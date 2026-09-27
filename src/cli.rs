//! Create-only JSONL command line for direct scoring, plus a stdin REPL.

use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Parser;
use serde_json::Value;

use crate::error::Error;
use crate::loader::load_model;
use crate::row::validate_row;

#[derive(Parser, Debug)]
#[command(
    name = "libre-semif-rs",
    about = "Score declared options from llama.cpp last-position logits"
)]
struct Args {
    /// Scoring mode. Only direct is implemented.
    #[arg(long)]
    mode: String,
    /// Hugging Face model id or a local directory with the tokenizer and chat template.
    #[arg(long)]
    model: String,
    /// 40-character commit for a remote model, or a revision label for a local directory.
    #[arg(long)]
    revision: String,
    /// Local GGUF checkpoint.
    #[arg(long)]
    gguf: PathBuf,
    /// Score JSON rows from stdin until EOF instead of reading --input.
    #[arg(long)]
    repl: bool,
    /// JSONL decisions. Each line is one row.
    #[arg(long, required_unless_present = "repl", conflicts_with = "repl")]
    input: Option<PathBuf>,
    /// New JSONL path. Existing files are refused.
    #[arg(long, required_unless_present = "repl", conflicts_with = "repl")]
    output: Option<PathBuf>,
    /// Maximum prompt tokens. Prompts are never truncated.
    #[arg(long, default_value_t = 4096)]
    max_tokens: i64,
    /// CPU threads for llama.cpp. Defaults to the visible core count.
    #[arg(long)]
    llama_threads: Option<i64>,
}

pub fn run_from<I, S>(args: I) -> ExitCode
where
    I: IntoIterator<Item = S>,
    S: Into<std::ffi::OsString> + Clone,
{
    let args = match Args::try_parse_from(args) {
        Ok(args) => args,
        Err(error) => {
            let code = u8::try_from(error.exit_code()).unwrap_or(2);
            let _ = error.print();
            return ExitCode::from(code);
        }
    };
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            if error.usage {
                ExitCode::from(2)
            } else {
                ExitCode::from(1)
            }
        }
    }
}

fn run(args: Args) -> Result<(), Error> {
    if args.mode != "direct" {
        return Err(Error::usage("--mode accepts only direct"));
    }
    if args.max_tokens < 1 {
        return Err(Error::usage(
            "Output must be new and max-tokens must be positive",
        ));
    }
    let threads = match args.llama_threads {
        Some(value) if value < 1 => return Err(Error::usage("--llama-threads must be positive")),
        Some(value) => Some(
            i32::try_from(value).map_err(|_| Error::usage("--llama-threads must be positive"))?,
        ),
        None => None,
    };
    if !args.gguf.is_file() {
        return Err(Error::usage(
            "--gguf must point at an existing GGUF file; requires --gguf",
        ));
    }
    let max_tokens =
        u32::try_from(args.max_tokens).map_err(|_| Error::usage("max-tokens is too large"))?;
    if args.repl {
        run_repl(&args, threads, max_tokens)
    } else {
        run_batch(&args, threads, max_tokens)
    }
}

fn run_batch(args: &Args, threads: Option<i32>, max_tokens: u32) -> Result<(), Error> {
    let input = args
        .input
        .as_ref()
        .ok_or_else(|| Error::usage("requires --input"))?;
    let output_path = args
        .output
        .as_ref()
        .ok_or_else(|| Error::usage("requires --output"))?;
    if output_path.exists() {
        return Err(Error::usage(
            "Output must be new and max-tokens must be positive",
        ));
    }
    let rows = read_rows(input)?;
    if rows.is_empty() {
        return Err(Error::new("Input is empty"));
    }
    for row in &rows {
        validate_row(row)?;
    }
    let mut session = load_model(&args.model, &args.revision, &args.gguf, threads, max_tokens)?;
    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|error| {
                Error::new(format!("Failed to create {}: {error}", parent.display()))
            })?;
        }
    }
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output_path)
        .map_err(|error| {
            Error::new(format!(
                "Output must be new and max-tokens must be positive: {error}"
            ))
        })?;
    for row in &rows {
        let result = session.score(row, max_tokens)?;
        let line = serde_json::to_string(&result)
            .map_err(|error| Error::new(format!("Failed to encode the result: {error}")))?;
        writeln!(output, "{line}")
            .map_err(|error| Error::new(format!("Failed to write the result: {error}")))?;
        output
            .flush()
            .map_err(|error| Error::new(format!("Failed to flush the result: {error}")))?;
    }
    Ok(())
}

fn run_repl(args: &Args, threads: Option<i32>, max_tokens: u32) -> Result<(), Error> {
    llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default().with_logs_enabled(false));
    let mut session = load_model(&args.model, &args.revision, &args.gguf, threads, max_tokens)?;
    let stdin = io::stdin();
    let stdout = io::stdout();
    let interactive = stdin.is_terminal();
    if interactive {
        eprintln!("Ready.");
    }
    repl_loop(stdin.lock(), stdout.lock(), interactive, |row| {
        validate_row(row)?;
        session.score(row, max_tokens)
    })
}

fn repl_loop<R, W, F>(reader: R, mut out: W, prompt: bool, mut score: F) -> Result<(), Error>
where
    R: Read,
    W: Write,
    F: FnMut(&Value) -> Result<Value, Error>,
{
    let mut stream = serde_json::Deserializer::from_reader(reader).into_iter::<Value>();
    loop {
        if prompt {
            eprint!("> ");
            io::stderr()
                .flush()
                .map_err(|error| Error::new(format!("Failed to write the prompt: {error}")))?;
        }
        let row = match stream.next() {
            None => return Ok(()),
            Some(Ok(row)) => row,
            Some(Err(error)) => {
                return Err(Error::new(format!("Input is not JSON: {error}")));
            }
        };
        let result = score(&row)?;
        let summary = format_summary(&result)?;
        write!(out, "{summary}")
            .map_err(|error| Error::new(format!("Failed to write the result: {error}")))?;
        out.flush()
            .map_err(|error| Error::new(format!("Failed to flush the result: {error}")))?;
    }
}

fn format_summary(result: &Value) -> Result<String, Error> {
    let ids = result
        .get("option_ids")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::new("Result is missing option scores"))?;
    let probabilities = result
        .get("probabilities")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::new("Result is missing option scores"))?;
    if ids.len() != probabilities.len() || ids.is_empty() {
        return Err(Error::new("Result is missing option scores"));
    }
    let mut ranked = Vec::with_capacity(ids.len());
    for (index, (id, probability)) in ids.iter().zip(probabilities).enumerate() {
        let id = id
            .as_str()
            .ok_or_else(|| Error::new("Result is missing option scores"))?;
        let probability = probability
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(|| Error::new("Result is missing option scores"))?;
        ranked.push((index, id, probability));
    }
    let width = ranked
        .iter()
        .map(|(_, id, _)| id.chars().count())
        .max()
        .unwrap_or(0);
    ranked.sort_by(|left, right| right.2.total_cmp(&left.2).then(left.0.cmp(&right.0)));
    let mut summary = String::new();
    for (_, id, probability) in ranked {
        let padding = " ".repeat(width - id.chars().count());
        summary.push_str(&format!("{id}{padding}  {probability:.4}\n"));
    }
    Ok(summary)
}

fn read_rows(path: &Path) -> Result<Vec<Value>, Error> {
    let file = File::open(path)
        .map_err(|error| Error::new(format!("Failed to read {}: {error}", path.display())))?;
    let mut rows = Vec::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line = line
            .map_err(|error| Error::new(format!("Failed to read {}: {error}", path.display())))?;
        if line.trim().is_empty() {
            continue;
        }
        let row = serde_json::from_str(&line).map_err(|error| {
            Error::new(format!("Input line {} is not JSON: {error}", index + 1))
        })?;
        rows.push(row);
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("libre-semif-rs-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn base_args(dir: &std::path::Path, extra: &[&str]) -> Vec<String> {
        let mut args = vec![
            "libre-semif-rs".to_string(),
            "--mode".to_string(),
            "direct".to_string(),
            "--model".to_string(),
            "Qwen/Qwen3.5-4B".to_string(),
            "--revision".to_string(),
            "a".repeat(40),
            "--gguf".to_string(),
            dir.join("missing.gguf").display().to_string(),
            "--input".to_string(),
            dir.join("in.jsonl").display().to_string(),
            "--output".to_string(),
            dir.join("out.jsonl").display().to_string(),
        ];
        args.extend(extra.iter().map(|item| (*item).to_string()));
        args
    }

    #[test]
    fn existing_output_is_refused_before_loading() {
        let dir = temp_dir("exists");
        let output = dir.join("out.jsonl");
        fs::write(&output, "keep\n").unwrap();
        let code = run_from(base_args(&dir, &[]));
        assert_eq!(code, ExitCode::from(2));
        assert_eq!(fs::read_to_string(&output).unwrap(), "keep\n");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn zero_max_tokens_and_threads_fail_before_loading() {
        let dir = temp_dir("flags");
        assert_eq!(
            run_from(base_args(&dir, &["--max-tokens", "0"])),
            ExitCode::from(2)
        );
        assert_eq!(
            run_from(base_args(&dir, &["--llama-threads", "0"])),
            ExitCode::from(2)
        );
        assert!(!dir.join("out.jsonl").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_gguf_file_is_rejected() {
        let dir = temp_dir("gguf");
        let code = run_from(base_args(&dir, &[]));
        assert_eq!(code, ExitCode::from(2));
        assert!(!dir.join("out.jsonl").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn other_modes_are_rejected() {
        let dir = temp_dir("mode");
        let mut args = base_args(&dir, &[]);
        let mode = args
            .iter_mut()
            .find(|item| item.as_str() == "direct")
            .unwrap();
        *mode = "serial".to_string();
        assert_eq!(run_from(args), ExitCode::from(2));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_input_fails_before_loading() {
        let dir = temp_dir("empty");
        fs::write(dir.join("in.jsonl"), "\n").unwrap();
        fs::write(dir.join("missing.gguf"), b"gguf").unwrap();
        let code = run_from(base_args(&dir, &[]));
        assert_eq!(code, ExitCode::from(1));
        assert!(!dir.join("out.jsonl").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    fn without(args: Vec<String>, flag: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut skip_next = false;
        for arg in args {
            if skip_next {
                skip_next = false;
                continue;
            }
            if arg == flag {
                skip_next = true;
                continue;
            }
            out.push(arg);
        }
        out
    }

    #[test]
    fn missing_output_is_usage() {
        let dir = temp_dir("no-output");
        assert_eq!(
            run_from(without(base_args(&dir, &[]), "--output")),
            ExitCode::from(2)
        );
        assert_eq!(
            run_from(without(base_args(&dir, &[]), "--input")),
            ExitCode::from(2)
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn repl_rejects_input_and_output() {
        let dir = temp_dir("repl-flags");
        assert_eq!(run_from(base_args(&dir, &["--repl"])), ExitCode::from(2));
        assert_eq!(
            run_from(without(base_args(&dir, &["--repl"]), "--output")),
            ExitCode::from(2)
        );
        assert_eq!(
            run_from(without(base_args(&dir, &["--repl"]), "--input")),
            ExitCode::from(2)
        );
        let _ = fs::remove_dir_all(&dir);
    }

    fn scored_row() -> Value {
        serde_json::json!({
            "option_ids": ["billing", "access"],
            "probabilities": [0.25, 0.75]
        })
    }

    #[test]
    fn repl_scores_a_pretty_printed_value_once() {
        let input = br#"{
  "id": "route-1",
  "state": "evidence"
}"#;
        let mut calls = 0;
        let mut out = Vec::new();
        repl_loop(&input[..], &mut out, false, |row| {
            calls += 1;
            assert_eq!(row["id"], "route-1");
            Ok(scored_row())
        })
        .unwrap();
        assert_eq!(calls, 1);
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "access   0.7500\nbilling  0.2500\n"
        );
    }

    #[test]
    fn repl_scores_two_values_on_one_line() {
        let input = br#"{"id":"a"}{"id":"b"}"#;
        let mut ids = Vec::new();
        let mut out = Vec::new();
        repl_loop(&input[..], &mut out, false, |row| {
            ids.push(row["id"].as_str().unwrap().to_string());
            Ok(scored_row())
        })
        .unwrap();
        assert_eq!(ids, ["a", "b"]);
        let text = String::from_utf8(out).unwrap();
        assert_eq!(text.matches("access   0.7500").count(), 2);
    }

    #[test]
    fn repl_invalid_json_prints_nothing() {
        let mut out = Vec::new();
        let error = repl_loop(&b"{"[..], &mut out, false, |_| unreachable!()).unwrap_err();
        assert!(out.is_empty());
        assert!(error.to_string().contains("JSON"));
        assert!(!error.usage);
    }

    #[test]
    fn repl_scorer_error_stops_the_loop() {
        let input = br#"{"id":"a"}{"id":"b"}{"id":"c"}"#;
        let mut calls = 0;
        let mut out = Vec::new();
        let error = repl_loop(&input[..], &mut out, false, |_| {
            calls += 1;
            if calls == 1 {
                Ok(scored_row())
            } else {
                Err(Error::new("score failed"))
            }
        })
        .unwrap_err();
        assert_eq!(calls, 2);
        assert!(error.to_string().contains("score failed"));
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "access   0.7500\nbilling  0.2500\n"
        );
    }

    #[test]
    fn repl_prompt_is_not_written_to_stdout() {
        let mut out = Vec::new();
        repl_loop(&br#"{"id":"a"}"#[..], &mut out, true, |_| Ok(scored_row())).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(!text.contains('>'));
        assert!(text.contains("access   0.7500"));
    }

    #[test]
    fn summary_sorts_descending_and_keeps_ties() {
        assert_eq!(
            format_summary(&scored_row()).unwrap(),
            "access   0.7500\nbilling  0.2500\n"
        );
        let tied = serde_json::json!({
            "option_ids": ["second", "first"],
            "probabilities": [0.5, 0.5]
        });
        assert_eq!(
            format_summary(&tied).unwrap(),
            "second  0.5000\nfirst   0.5000\n"
        );
    }
}
