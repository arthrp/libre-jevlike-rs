//! Row validation, the direct-mode prompt, and numeric helpers.

use std::collections::HashSet;

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::error::Error;

pub const LETTERS: &str = "ABCDEFGHIJKLMNOP";
pub const DIRECT_SYSTEM: &str = "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning.";

const REQUIRED: [&str; 4] = ["id", "options", "question", "state"];

pub fn validate_row(row: &Value) -> Result<(), Error> {
    let Some(object) = row.as_object() else {
        return Err(Error::new(format!(
            "Row is missing fields: {}",
            REQUIRED.join(", ")
        )));
    };
    let missing: Vec<&str> = REQUIRED
        .into_iter()
        .filter(|key| !object.contains_key(*key))
        .collect();
    if !missing.is_empty() {
        return Err(Error::new(format!(
            "Row is missing fields: {}",
            missing.join(", ")
        )));
    }
    for key in ["id", "question"] {
        match object.get(key) {
            Some(Value::String(text)) if !text.is_empty() => {}
            _ => return Err(Error::new("id and question must be nonempty strings")),
        }
    }
    match object.get("state") {
        Some(Value::String(text)) if !text.is_empty() => {}
        Some(Value::Object(map)) if !map.is_empty() => {}
        Some(Value::Array(items)) if !items.is_empty() => {}
        _ => {
            return Err(Error::new(
                "state must be a nonempty string, object, or array",
            ))
        }
    }
    let options = match object.get("options") {
        Some(Value::Array(items)) if (2..=LETTERS.chars().count()).contains(&items.len()) => items,
        _ => return Err(Error::new("options must contain 2-16 entries")),
    };
    let mut ids = Vec::with_capacity(options.len());
    for option in options {
        let Some(option) = option.as_object() else {
            return Err(Error::new(
                "Each option needs string id and description fields",
            ));
        };
        match (option.get("id"), option.get("description")) {
            (Some(Value::String(id)), Some(Value::String(_))) => ids.push(id.as_str()),
            _ => {
                return Err(Error::new(
                    "Each option needs string id and description fields",
                ))
            }
        }
    }
    if ids.len() != ids.iter().collect::<HashSet<_>>().len() {
        return Err(Error::new("Option IDs must be unique"));
    }
    Ok(())
}

pub fn direct_messages(row: &Value) -> Result<Vec<Value>, Error> {
    validate_row(row)?;
    let object = row.as_object().expect("validate_row requires an object");
    let options = object["options"].as_array().expect("options is an array");
    let mut rendered_options = Vec::with_capacity(options.len());
    for (index, option) in options.iter().enumerate() {
        let letter = LETTERS.chars().nth(index).expect("index is in range");
        let mut item = Map::new();
        item.insert("letter".to_string(), Value::String(letter.to_string()));
        item.insert("description".to_string(), option["description"].clone());
        rendered_options.push(Value::Object(item));
    }
    let mut payload = Map::new();
    payload.insert("evidence".to_string(), object["state"].clone());
    payload.insert("criterion".to_string(), object["question"].clone());
    payload.insert("options".to_string(), Value::Array(rendered_options));
    let content = serde_json::to_string(&Value::Object(payload))
        .map_err(|error| Error::new(format!("could not serialize prompt payload: {error}")))?;
    Ok(vec![
        message("system", DIRECT_SYSTEM),
        message("user", &content),
    ])
}

fn message(role: &str, content: &str) -> Value {
    let mut item = Map::new();
    item.insert("role".to_string(), Value::String(role.to_string()));
    item.insert("content".to_string(), Value::String(content.to_string()));
    Value::Object(item)
}

pub fn softmax(values: &[f64]) -> Result<Vec<f64>, Error> {
    if values.len() < 2 || values.iter().any(|value| !value.is_finite()) {
        return Err(Error::new("Need at least two finite scores"));
    }
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = values.iter().map(|value| (value - maximum).exp()).collect();
    let total: f64 = weights.iter().sum();
    Ok(weights.into_iter().map(|weight| weight / total).collect())
}

pub fn digest(text: &str) -> String {
    hex::encode(Sha256::digest(text.as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn row() -> Value {
        json!({
            "id": "x",
            "state": "owned evidence",
            "question": "Which answer follows?",
            "options": [
                {"id": "yes", "description": "Yes."},
                {"id": "no", "description": "No."}
            ]
        })
    }

    #[test]
    fn direct_prompt_excludes_extra_fields() {
        let mut value = row();
        value["label"] = json!("yes");
        value["provenance"] = json!({"secret": "do not leak"});
        let rendered = format!("{:?}", direct_messages(&value).unwrap());
        assert!(rendered.contains("owned evidence"));
        assert!(!rendered.contains("secret"));
        assert!(!rendered.contains("label"));
    }

    #[test]
    fn softmax_is_finite_and_normalized() {
        let values = softmax(&[1000.0, 999.0, -1000.0]).unwrap();
        assert!(values.iter().all(|value| value.is_finite()));
        assert!((values.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(values[0] > values[1] && values[1] > values[2]);
    }

    #[test]
    fn softmax_rejects_too_few_or_non_finite() {
        assert!(softmax(&[1.0]).unwrap_err().to_string().contains("finite"));
        assert!(softmax(&[1.0, f64::NAN]).is_err());
    }

    #[test]
    fn duplicate_options_rejected() {
        let mut value = row();
        value["options"] = json!([
            {"id": "yes", "description": "Yes."},
            {"id": "yes", "description": "Yes."}
        ]);
        let error = validate_row(&value).unwrap_err();
        assert!(error.to_string().contains("unique"));
    }

    #[test]
    fn structured_json_state_is_supported() {
        let mut value = row();
        value["state"] = json!({"policy": "Never request passwords", "candidate": ["invoice id"]});
        validate_row(&value).unwrap();
        let messages = direct_messages(&value).unwrap();
        assert!(messages[1]["content"]
            .as_str()
            .unwrap()
            .contains("\"policy\""));
    }

    #[test]
    fn missing_fields_are_listed() {
        let error = validate_row(&json!({"id": "x"})).unwrap_err();
        assert_eq!(
            error.to_string(),
            "Row is missing fields: options, question, state"
        );
    }

    #[test]
    fn user_message_is_compact_json() {
        let messages = direct_messages(&row()).unwrap();
        assert_eq!(
            messages[1]["content"].as_str().unwrap(),
            r#"{"evidence":"owned evidence","criterion":"Which answer follows?","options":[{"letter":"A","description":"Yes."},{"letter":"B","description":"No."}]}"#
        );
    }

    #[test]
    fn empty_state_and_short_options_are_rejected() {
        let mut empty_state = row();
        empty_state["state"] = json!("");
        assert!(validate_row(&empty_state)
            .unwrap_err()
            .to_string()
            .contains("nonempty"));
        let mut one_option = row();
        one_option["options"] = json!([{"id": "yes", "description": "Yes."}]);
        assert!(validate_row(&one_option)
            .unwrap_err()
            .to_string()
            .contains("2-16"));
    }
}
