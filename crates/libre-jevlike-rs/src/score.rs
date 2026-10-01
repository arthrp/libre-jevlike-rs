//! Direct-mode scoring and the llama.cpp JSONL record.

use std::time::Instant;

use serde_json::{json, Map, Value};

use crate::engine::full_logits;
use crate::error::Error;
use crate::loader::Session;
use crate::prompt::{encode_prompt, PROMPT_VERSION};
use crate::row::softmax;

const READOUT: &str =
    "quantized last-position logits restricted to declared answer slots; no generated tokens";
const STATUS_QUANTIZED: &str =
    "conditional option score over quantized weights; uncalibrated as decision confidence";
const STATUS_FULL: &str = "conditional option score; uncalibrated as decision confidence";

impl Session {
    pub fn score(&mut self, row: &Value, max_tokens: u32) -> Result<Value, Error> {
        let started = Instant::now();
        let encoded = encode_prompt(
            self.tokenizer(),
            row,
            usize::try_from(max_tokens).unwrap_or(usize::MAX),
        )?;
        let gguf_ids = self.gguf_tokenize(&encoded.prompt)?;
        if gguf_ids != encoded.ids {
            let row_id = row["id"].as_str().unwrap_or("");
            return Err(Error::new(format!(
                "Row {row_id}: GGUF tokenization disagrees with the reference tokenizer"
            )));
        }
        let mark = Instant::now();
        let vocabulary = full_logits(self.context(), &encoded.ids)?;
        let selected = selected_logits(&vocabulary, &encoded.slots)?;
        let probabilities = softmax(&selected)?;
        let allowed_token_mass = (logsumexp(&selected) - logsumexp_f32(&vocabulary)).exp();
        let full_vocab_argmax_id = argmax(&vocabulary)?;
        let forward_seconds = mark.elapsed().as_secs_f64();
        let total_seconds = started.elapsed().as_secs_f64();
        let option_ids = row["options"]
            .as_array()
            .ok_or_else(|| Error::new("options must contain 2-16 entries"))?
            .iter()
            .map(|option| option["id"].clone())
            .collect::<Vec<_>>();
        let mut model = self.metadata.as_object().cloned().unwrap_or_default();
        model.insert(
            "serving_config".to_string(),
            json!(format!(
                "llamacpp-{}-direct-v1",
                self.scoring_backend().name()
            )),
        );
        let status = if self.quantized() {
            STATUS_QUANTIZED
        } else {
            STATUS_FULL
        };
        let mut result = Map::new();
        result.insert("id".to_string(), row["id"].clone());
        result.insert("option_ids".to_string(), Value::Array(option_ids));
        result.insert("probabilities".to_string(), f64s(&probabilities)?);
        result.insert("option_logits".to_string(), f64s(&selected)?);
        result.insert("answer_token_ids".to_string(), json!(encoded.slots));
        result.insert("input_tokens".to_string(), json!(encoded.ids.len()));
        result.insert(
            "allowed_token_mass".to_string(),
            finite_f64(allowed_token_mass)?,
        );
        result.insert(
            "full_vocab_argmax_id".to_string(),
            json!(full_vocab_argmax_id),
        );
        result.insert("prompt_sha256".to_string(), json!(encoded.prompt_sha256));
        result.insert("prompt_version".to_string(), json!(PROMPT_VERSION));
        result.insert("model".to_string(), Value::Object(model));
        result.insert("readout".to_string(), json!(READOUT));
        result.insert("probability_status".to_string(), json!(status));
        result.insert("forward_seconds".to_string(), finite_f64(forward_seconds)?);
        result.insert("total_seconds".to_string(), finite_f64(total_seconds)?);
        Ok(Value::Object(result))
    }
}

fn selected_logits(vocabulary: &[f32], slots: &[i32]) -> Result<Vec<f64>, Error> {
    let mut selected = Vec::with_capacity(slots.len());
    for slot in slots {
        let index = usize::try_from(*slot)
            .map_err(|_| Error::new(format!("Answer token {slot} is out of range")))?;
        let logit = vocabulary
            .get(index)
            .copied()
            .ok_or_else(|| Error::new(format!("Answer token {slot} is outside the vocabulary")))?;
        if !logit.is_finite() {
            return Err(Error::new("logits were not finite"));
        }
        selected.push(f64::from(logit));
    }
    Ok(selected)
}

fn logsumexp(values: &[f64]) -> f64 {
    let peak = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !peak.is_finite() {
        return peak;
    }
    let sum: f64 = values.iter().map(|value| (value - peak).exp()).sum();
    peak + sum.ln()
}

fn logsumexp_f32(values: &[f32]) -> f64 {
    let peak = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !peak.is_finite() {
        return f64::from(peak);
    }
    let peak = f64::from(peak);
    let sum: f64 = values
        .iter()
        .map(|value| (f64::from(*value) - peak).exp())
        .sum();
    peak + sum.ln()
}

fn argmax(values: &[f32]) -> Result<i64, Error> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(Error::new("logits were not finite"));
    }
    let mut best = 0usize;
    for (index, value) in values.iter().enumerate().skip(1) {
        if *value > values[best] {
            best = index;
        }
    }
    Ok(i64::try_from(best).unwrap_or(0))
}

fn f64s(values: &[f64]) -> Result<Value, Error> {
    let mut items = Vec::with_capacity(values.len());
    for value in values {
        items.push(finite_f64(*value)?);
    }
    Ok(Value::Array(items))
}

fn finite_f64(value: f64) -> Result<Value, Error> {
    serde_json::Number::from_f64(value)
        .map(Value::Number)
        .ok_or_else(|| Error::new("score was not finite"))
}
