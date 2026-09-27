# libre-semif-rs

This is Jev-like semantic decision service, inspired by [SemIf-OpenJev](https://github.com/TheoLeeCJ/SemIf-OpenJev). The prompt layout and row checks are based on that project's direct mode. This crate implements only that path through llama.cpp: Metal on Apple Silicon, and Vulkan or CPU on Linux x86_64.

`libre-semif-rs` reads JSONL decisions and writes a new JSONL of option probabilities. No answer text is generated.

## Usage

On macOS the host must be Apple Silicon and every layer is offloaded to Metal. On Linux x86_64 the default build offloads every layer to Vulkan and refuses to run without a Vulkan GPU. A CPU-only Linux build runs every layer on the CPU. `--mode` accepts only `direct`. The output file must not already exist. Prompts are never truncated; `--max-tokens` (default 4096) is a hard limit.

```bash
libre-semif-rs \
  --mode direct \
  --model Qwen/Qwen3.5-4B \
  --revision <40-character-commit> \
  --gguf /path/to/model.gguf \
  --input decisions.jsonl \
  --output results.jsonl
```

`--llama-threads` sets llama.cpp CPU threads. A remote `--model` needs a 40-character revision; a local directory needs a revision label and the tokenizer plus chat template.

### REPL

`--repl` reads JSON rows from stdin and prints a probability summary. `--input` and `--output` cannot be used with it. A row can be one line or pretty-printed across several lines; each value is scored as soon as it is complete. The model stays loaded until stdin ends.

```bash
libre-semif-rs \
  --repl \
  --mode direct \
  --model Qwen/Qwen3.5-4B \
  --revision <40-character-commit> \
  --gguf /path/to/model.gguf
```

Each row prints one line per option, `option_id` then probability, highest first. Ctrl-D (or the end of a pipe) exits. Invalid JSON, a row that fails the usual checks, or a scoring error stops the process. llama.cpp's own logs are silenced in this mode. When stdin is a terminal, a prompt is written to stderr so stdout stays limited to the summary.

Each input line is one row: `id`, `state` (nonempty string, object, or array), `question`, and 2–16 options with unique `id` and `description`.

```json
{"id":"route-1","state":"Customer cannot access an account after a password reset.","question":"Which queue should handle this request?","options":[{"id":"access","description":"Account access support."},{"id":"billing","description":"Billing support."}]}
```

Each result line includes `probabilities`, `option_logits`, `prompt_sha256`, `prompt_version`, and model metadata (source, revision, GGUF checksum, backend). Probabilities are conditional on the declared options and uncalibrated as decision confidence.

## Building

macOS uses Metal. The `vulkan` and `cpu` features apply on Linux; a macOS build uses Metal either way.

```bash
cargo build --release
```

Linux x86_64 uses Vulkan by default. That build needs CMake, a C++ toolchain, the Vulkan headers and loader, `glslc`, and libgomp:

```bash
cargo build --release
```

CPU-only, still with OpenMP. This build needs CMake, a C++ toolchain, and libgomp:

```bash
cargo build --release --no-default-features --features cpu
```

## Scoring

1. Build a system message plus a user payload of evidence, criterion, and options labeled `A`–`P`.
2. Render the checkpoint chat template and tokenize with the pinned Hugging Face tokenizer. Each letter must be one token that still appends cleanly. GGUF tokenization must match.
3. One llama.cpp decode of the prompt, in 512-token chunks. Logits are requested only at the last position.
4. Keep the logits for those letter slots, softmax them into probabilities, and record `allowed_token_mass` and `full_vocab_argmax_id`.

## Layout

- `cli` — JSONL in, create-only JSONL out, or `--repl` from stdin.
- `loader` — pinned tokenizer and local GGUF, one llama.cpp context.
- `prompt` — chat-template render and answer-slot checks (`direct-options-v2`).
- `engine` — last-position logits; no generated tokens.
- `score` — slot logits, softmax, and the result record.
- `row` — row validation and the direct-mode messages.
- `gguf` — Qwen3 / Qwen3.5 architecture and file-type metadata only.
