//! Chat-template rendering and the direct-mode prompt contract.
//!
//! The checkpoint's own Jinja template is rendered with MiniJinja. Answer slots
//! must be single tokens that survive appending the letter to the prompt, matching
//! `encode_prompt` in `src/semif_phase1/direct.py`.

use std::path::Path;

use minijinja::value::Value as JinjaValue;
use minijinja::{context, Environment, ErrorKind};
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::error::Error;
use crate::row::{direct_messages, LETTERS};

pub const PROMPT_VERSION: &str = "direct-options-v2";

#[derive(Debug)]
pub struct EncodedPrompt {
    pub prompt: String,
    pub ids: Vec<i32>,
    pub slots: Vec<i32>,
    pub prompt_sha256: String,
}

pub(crate) trait PromptTokenizer {
    fn render(&self, messages: &[Value]) -> Result<String, Error>;
    fn encode(&self, text: &str) -> Result<Vec<i32>, Error>;
    fn decode(&self, ids: &[i32]) -> Result<String, Error>;
}

pub struct ReferenceTokenizer {
    tokenizer: Tokenizer,
    env: Environment<'static>,
}

impl ReferenceTokenizer {
    pub fn from_files(tokenizer_json: &Path, template: &str) -> Result<Self, Error> {
        let mut tokenizer = Tokenizer::from_file(tokenizer_json).map_err(|error| {
            Error::new(format!(
                "Failed to load tokenizer {}: {error}",
                tokenizer_json.display()
            ))
        })?;
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(|error| Error::new(format!("Failed to disable truncation: {error}")))?;
        Ok(Self {
            tokenizer,
            env: chat_environment(template)?,
        })
    }
}

impl PromptTokenizer for ReferenceTokenizer {
    fn render(&self, messages: &[Value]) -> Result<String, Error> {
        let template = self
            .env
            .get_template("chat")
            .map_err(|error| Error::new(format!("Chat template failed to load: {error}")))?;
        let messages = JinjaValue::from_serialize(messages);
        template
            .render(context! {
                messages => messages,
                add_generation_prompt => true,
                enable_thinking => false,
            })
            .map_err(|error| Error::new(format!("Chat template failed: {error}")))
    }

    fn encode(&self, text: &str) -> Result<Vec<i32>, Error> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|error| Error::new(format!("Tokenizer rejected the prompt: {error}")))?;
        encoding
            .get_ids()
            .iter()
            .map(|id| {
                i32::try_from(*id)
                    .map_err(|_| Error::new(format!("Token id {id} does not fit in i32")))
            })
            .collect()
    }

    fn decode(&self, ids: &[i32]) -> Result<String, Error> {
        let ids = ids
            .iter()
            .map(|id| {
                u32::try_from(*id).map_err(|_| Error::new(format!("Token id {id} is negative")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.tokenizer
            .decode(&ids, false)
            .map_err(|error| Error::new(format!("Tokenizer failed to decode a token: {error}")))
    }
}

pub fn encode_prompt(
    tokenizer: &impl PromptTokenizer,
    row: &Value,
    max_tokens: usize,
) -> Result<EncodedPrompt, Error> {
    let messages = direct_messages(row)?;
    let prompt = tokenizer.render(&messages)?;
    let ids = tokenizer.encode(&prompt)?;
    let row_id = row["id"].as_str().unwrap_or("");
    if ids.is_empty() || ids.len() > max_tokens {
        return Err(Error::new(format!(
            "Row {row_id}: {} input tokens exceed limit {max_tokens}; no truncation allowed",
            ids.len()
        )));
    }
    let slots = slot_ids(
        tokenizer,
        row["options"].as_array().map(Vec::len).unwrap_or(0),
    )?;
    for (letter, token) in LETTERS.chars().zip(slots.iter().copied()) {
        let extended = tokenizer.encode(&format!("{prompt}{letter}"))?;
        let mut expected = ids.clone();
        expected.push(token);
        if extended != expected {
            return Err(Error::new(format!(
                "Answer boundary changes tokenization for slot {}",
                python_repr(&letter.to_string())
            )));
        }
    }
    let prompt_sha256 = crate::row::digest(&prompt);
    Ok(EncodedPrompt {
        prompt,
        ids,
        slots,
        prompt_sha256,
    })
}

fn slot_ids(tokenizer: &impl PromptTokenizer, count: usize) -> Result<Vec<i32>, Error> {
    let mut result = Vec::with_capacity(count);
    for letter in LETTERS.chars().take(count) {
        let letter = letter.to_string();
        let encoded = tokenizer.encode(&letter)?;
        let decoded = tokenizer.decode(&encoded)?;
        if encoded.len() != 1 || decoded != letter {
            return Err(Error::new(format!(
                "Answer slot {} is not one exact round-trip token",
                python_repr(&letter)
            )));
        }
        result.push(encoded[0]);
    }
    if result.len()
        != result
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
    {
        return Err(Error::new("Answer-slot tokens collide"));
    }
    Ok(result)
}

pub fn python_repr(text: &str) -> String {
    format!("'{}'", text.replace('\\', "\\\\").replace('\'', "\\'"))
}

fn chat_environment(template: &str) -> Result<Environment<'static>, Error> {
    let mut env = Environment::new();
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function("raise_exception", raise_exception);
    let _ = env.add_test("false", is_false);
    let _ = env.add_test("true", is_true);
    let _ = env.add_test("iterable", is_iterable);
    env.add_filter("items", items_filter);
    env.add_template_owned("chat", template.to_string())
        .map_err(|error| Error::new(format!("Chat template failed to compile: {error}")))?;
    Ok(env)
}

fn raise_exception(message: String) -> Result<JinjaValue, minijinja::Error> {
    Err(minijinja::Error::new(ErrorKind::InvalidOperation, message))
}

fn is_bool(value: &JinjaValue, expected: bool) -> bool {
    value.kind() == minijinja::value::ValueKind::Bool && value.is_true() == expected
}

fn is_false(value: JinjaValue) -> bool {
    is_bool(&value, false)
}

fn is_true(value: JinjaValue) -> bool {
    is_bool(&value, true)
}

fn is_iterable(value: JinjaValue) -> bool {
    value.try_iter().is_ok()
}

fn items_filter(value: JinjaValue) -> Result<JinjaValue, minijinja::Error> {
    let mut pairs = Vec::new();
    for key in value.try_iter()? {
        let item = value.get_item(&key)?;
        pairs.push(JinjaValue::from(vec![key, item]));
    }
    Ok(JinjaValue::from(pairs))
}

pub fn chat_template_from_config(config: &Value) -> Result<String, Error> {
    let template = config.get("chat_template").ok_or_else(|| {
        Error::new("tokenizer_config.json has no chat_template and chat_template.jinja is absent")
    })?;
    match template {
        Value::String(text) => Ok(text.clone()),
        Value::Array(items) => items
            .iter()
            .find(|item| item.get("name").and_then(Value::as_str) == Some("default"))
            .or_else(|| items.first())
            .and_then(|item| item.get("template").and_then(Value::as_str))
            .map(str::to_string)
            .ok_or_else(|| Error::new("chat_template list has no template string")),
        Value::Object(map) => map
            .get("default")
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| Error::new("chat_template object has no default string")),
        _ => Err(Error::new("chat_template must be a string")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct ByteTokenizer;

    impl PromptTokenizer for ByteTokenizer {
        fn render(&self, messages: &[Value]) -> Result<String, Error> {
            let mut out = String::new();
            for message in messages {
                out.push_str(message["content"].as_str().unwrap_or(""));
                out.push('\n');
            }
            out.push_str("Assistant:");
            Ok(out)
        }

        fn encode(&self, text: &str) -> Result<Vec<i32>, Error> {
            Ok(text.bytes().map(i32::from).collect())
        }

        fn decode(&self, ids: &[i32]) -> Result<String, Error> {
            let bytes = ids
                .iter()
                .map(|id| u8::try_from(*id).map_err(|_| Error::new("token is not a byte")))
                .collect::<Result<Vec<_>, _>>()?;
            String::from_utf8(bytes).map_err(|error| Error::new(error.to_string()))
        }
    }

    #[test]
    fn byte_tokenizer_accepts_answer_slots() {
        let row = json!({
            "id": "x",
            "state": "owned evidence",
            "question": "Which answer follows?",
            "options": [
                {"id": "yes", "description": "Yes."},
                {"id": "no", "description": "No."}
            ]
        });
        let encoded = encode_prompt(&ByteTokenizer, &row, 4096).unwrap();
        assert_eq!(encoded.slots, vec![i32::from(b'A'), i32::from(b'B')]);
        assert!(encoded.prompt.contains("owned evidence"));
        assert!(encoded.prompt.ends_with("Assistant:"));
        assert_eq!(encoded.prompt_sha256.len(), 64);
    }

    #[test]
    fn token_limit_is_enforced() {
        let row = json!({
            "id": "x",
            "state": "owned evidence",
            "question": "Which answer follows?",
            "options": [
                {"id": "yes", "description": "Yes."},
                {"id": "no", "description": "No."}
            ]
        });
        let error = encode_prompt(&ByteTokenizer, &row, 4).unwrap_err();
        assert!(error.to_string().contains("no truncation allowed"));
    }

    #[test]
    fn thinking_flag_renders_closed_think_block() {
        let env = chat_environment(
            "{%- if enable_thinking is defined and enable_thinking is false -%}off{%- else -%}on{%- endif -%}",
        )
        .unwrap();
        let template = env.get_template("chat").unwrap();
        let rendered = template
            .render(context! { enable_thinking => false })
            .unwrap();
        assert_eq!(rendered, "off");
    }

    #[test]
    fn qwen_template_matches_transformers_when_present() {
        let Some(snapshot) = qwen_snapshot() else {
            eprintln!("skipping Qwen template comparison; snapshot not in the Hugging Face cache");
            return;
        };
        let template = std::fs::read_to_string(snapshot.join("chat_template.jinja")).unwrap();
        let reference =
            ReferenceTokenizer::from_files(&snapshot.join("tokenizer.json"), &template).unwrap();
        let row = json!({
            "id": "support-1",
            "state": "The deployment completed at 14:02 UTC. Health checks passed in all three zones. No rollback was initiated.",
            "question": "Is there evidence that the deployment succeeded?",
            "options": [
                {"id": "yes", "description": "The deployment succeeded."},
                {"id": "no", "description": "The deployment did not succeed."},
                {"id": "insufficient", "description": "The evidence is insufficient to decide."}
            ]
        });
        let messages = direct_messages(&row).unwrap();
        let prompt = reference.render(&messages).unwrap();
        let ids = reference.encode(&prompt).unwrap();
        assert!(prompt.contains("<|im_start|>system\n"));
        assert!(prompt.contains("<think>\n\n</think>\n\n"));
        assert!(prompt.contains("deployment completed"));
        let Some(python) = venv_python() else {
            return;
        };
        let messages_json = serde_json::to_string(&messages).unwrap();
        let output = std::process::Command::new(python)
            .arg("-c")
            .arg(PYTHON_TEMPLATE)
            .arg(&snapshot)
            .arg(messages_json)
            .output()
            .expect("python");
        if !output.status.success() {
            panic!(
                "transformers comparison failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let expected: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(prompt, expected["prompt"].as_str().unwrap());
        let expected_ids: Vec<i32> = expected["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_i64().unwrap() as i32)
            .collect();
        assert_eq!(ids, expected_ids);
    }

    fn qwen_snapshot() -> Option<std::path::PathBuf> {
        if let Ok(path) = std::env::var("SEMIF_HF_TOKENIZER") {
            let path = std::path::PathBuf::from(path);
            return path.is_dir().then_some(path);
        }
        let home = std::env::var_os("HOME")?;
        let path = std::path::PathBuf::from(home).join(
            ".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        );
        path.is_dir().then_some(path)
    }

    fn venv_python() -> Option<std::path::PathBuf> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../.venv/bin/python");
        path.is_file().then_some(path)
    }

    const PYTHON_TEMPLATE: &str = r#"
import json, sys
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True, trust_remote_code=False)
messages = json.loads(sys.argv[2])
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tokenizer.encode(prompt, add_special_tokens=False)
json.dump({"prompt": prompt, "ids": ids}, sys.stdout)
"#;
}
