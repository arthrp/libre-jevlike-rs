//! Opt-in check against a local GGUF. Set `SEMIF_LLAMACPP_GGUF` to run it.

use std::path::PathBuf;

use libre_jevlike_rs::{compiled_backend_name, load_model};
use serde_json::json;

#[test]
fn real_gguf_scores_direct() {
    let Ok(gguf) = std::env::var("SEMIF_LLAMACPP_GGUF") else {
        eprintln!("skipping real GGUF test; set SEMIF_LLAMACPP_GGUF to a local GGUF path");
        return;
    };
    let source =
        std::env::var("SEMIF_LLAMACPP_SOURCE").unwrap_or_else(|_| "Qwen/Qwen3.5-4B".to_string());
    let revision = std::env::var("SEMIF_LLAMACPP_REVISION")
        .unwrap_or_else(|_| "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a".to_string());
    let threads = std::env::var("SEMIF_LLAMACPP_THREADS")
        .ok()
        .and_then(|value| value.parse::<i32>().ok())
        .unwrap_or(8);
    let state = "The deployment completed at 14:02 UTC. Health checks passed in all three zones.";
    let rows = [
        json!({
            "id": "deployment",
            "state": state,
            "question": "Is there evidence that the deployment succeeded?",
            "options": [
                {"id": "yes", "description": "The deployment succeeded."},
                {"id": "no", "description": "The deployment did not succeed."}
            ]
        }),
        json!({
            "id": "health",
            "state": state,
            "question": "Did the health checks pass?",
            "options": [
                {"id": "yes", "description": "The checks passed."},
                {"id": "no", "description": "The checks failed."}
            ]
        }),
    ];
    let mut session = load_model(
        &source,
        &revision,
        PathBuf::from(gguf).as_path(),
        Some(threads),
        4096,
    )
    .expect("load GGUF");
    let metadata = session.metadata.clone();
    let results = rows
        .iter()
        .map(|row| session.score(row, 4096).expect("score"))
        .collect::<Vec<_>>();
    let choices = results
        .iter()
        .map(|result| {
            let probabilities = result["probabilities"].as_array().unwrap();
            let mut best = 0usize;
            for (index, value) in probabilities.iter().enumerate().skip(1) {
                if value.as_f64().unwrap() > probabilities[best].as_f64().unwrap() {
                    best = index;
                }
            }
            result["option_ids"][best].as_str().unwrap().to_string()
        })
        .collect::<Vec<_>>();
    assert_eq!(choices, ["yes", "yes"]);
    let backend_name = compiled_backend_name();
    assert_eq!(metadata["backend"], format!("llamacpp-{backend_name}"));
    if backend_name == "cpu" {
        assert_eq!(metadata["n_gpu_layers"], 0);
    } else {
        assert!(metadata["n_gpu_layers"].as_u64().unwrap() >= 1);
    }
    assert_eq!(metadata["max_prompt_tokens"], 4096);
    assert!(
        metadata["context_tokens"].as_u64().unwrap()
            >= metadata["max_prompt_tokens"].as_u64().unwrap()
    );
    assert_eq!(metadata["decode_chunk"], 512);
}
