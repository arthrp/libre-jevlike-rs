//! Load a pinned Hugging Face tokenizer and a local GGUF checkpoint.

use std::fs::File;
use std::io::Read;
use std::mem::ManuallyDrop;
use std::num::NonZeroU32;
use std::path::Path;
use std::thread;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::LlamaToken;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::engine::recorded_context_tokens;
use crate::error::Error;
use crate::gguf::{self, dtype_name};
use crate::prompt::{chat_template_from_config, python_repr, PromptTokenizer, ReferenceTokenizer};
use crate::row::{direct_messages, LETTERS};

pub const LLAMA_CPP_2_VERSION: &str = "0.1.157";
pub const TOKENIZERS_VERSION: &str = "0.23.2";
pub const MINIJINJA_VERSION: &str = "2.24.0";

/// A loaded GGUF model, its reference tokenizer, and one llama.cpp context.
///
/// `LlamaContext` borrows the model. The model is boxed so that address stays
/// stable when this value is moved. The context is freed before the model, and
/// the backend is freed last.
pub struct Session {
    context: ManuallyDrop<LlamaContext<'static>>,
    model: Box<LlamaModel>,
    /// Process-wide llama.cpp backend. Stored so it outlives the context; Drop frees the context first.
    #[allow(dead_code)]
    backend: LlamaBackend,
    scoring: Backend,
    tokenizer: ReferenceTokenizer,
    pub metadata: Value,
    quantized: bool,
}

impl Drop for Session {
    fn drop(&mut self) {
        // Safety: this is the only place the context is dropped, and it runs
        // before `model` and `backend` are dropped.
        unsafe { ManuallyDrop::drop(&mut self.context) }
    }
}

impl Session {
    pub(crate) fn context(&mut self) -> &mut LlamaContext<'static> {
        &mut self.context
    }

    pub(crate) fn tokenizer(&self) -> &ReferenceTokenizer {
        &self.tokenizer
    }

    pub(crate) fn quantized(&self) -> bool {
        self.quantized
    }

    pub(crate) fn gguf_tokenize(&self, text: &str) -> Result<Vec<i32>, Error> {
        tokenize_with(&self.model, text)
    }

    pub(crate) fn scoring_backend(&self) -> Backend {
        self.scoring
    }
}

pub fn load_model(
    source: &str,
    revision: &str,
    gguf: &Path,
    threads: Option<i32>,
    context_tokens: u32,
) -> Result<Session, Error> {
    let local = check_revision(source, revision)?;
    if !gguf.is_file() {
        return Err(Error::new(format!(
            "GGUF checkpoint not found: {}",
            gguf.display()
        )));
    }
    let threads = resolve_threads(threads)?;
    if context_tokens == 0 {
        return Err(Error::new("context_tokens must be a positive integer"));
    }
    let scoring = compiled_backend();
    if !host_backend_supported(std::env::consts::OS, std::env::consts::ARCH, scoring) {
        return Err(unsupported_host(scoring));
    }
    // Hash and parse before any llama.cpp allocation.
    let (sha256, bytes) = hash_file(gguf)?;
    let info = gguf::read_info(gguf)?;
    let tokenizer = load_tokenizer(source, revision, local)?;

    let backend = LlamaBackend::init()
        .map_err(|error| Error::new(format!("llama.cpp backend failed to initialize: {error}")))?;
    require_offload(&backend, scoring)?;
    let params = model_params(scoring);
    let model = LlamaModel::load_from_file(&backend, gguf, &params).map_err(|error| {
        Error::new(format!(
            "llama.cpp failed to load the GGUF checkpoint {}: {error}",
            gguf.display()
        ))
    })?;
    verify_vocabulary(&model, &tokenizer)?;

    let window = context_tokens
        .checked_add(64)
        .ok_or_else(|| Error::new("context_tokens is too large"))?;
    let n_batch = window.min(2048);
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(window))
        .with_n_batch(n_batch)
        .with_n_ubatch(n_batch.min(512).max(1))
        .with_n_seq_max(1)
        .with_n_threads(threads)
        .with_n_threads_batch(threads);
    let model = Box::new(model);
    let context = model.new_context(&backend, ctx_params).map_err(|error| {
        Error::new(format!(
            "llama.cpp failed to create the scoring context: {error}"
        ))
    })?;
    let actual_context = recorded_context_tokens(context.n_ctx());
    // Safety: `context` borrows `*model`. The model is on the heap, so moving
    // the box does not move it. `Session`'s Drop frees the context first.
    let context = ManuallyDrop::new(unsafe {
        std::mem::transmute::<LlamaContext<'_>, LlamaContext<'static>>(context)
    });
    let file_name = gguf
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model.gguf");
    let mut metadata = Map::new();
    metadata.insert("source".to_string(), json!(source));
    metadata.insert("revision".to_string(), json!(revision));
    metadata.insert(
        "backend".to_string(),
        json!(format!("llamacpp-{}", scoring.name())),
    );
    metadata.insert("dtype".to_string(), json!(dtype_name(&info)));
    metadata.insert(
        "gguf".to_string(),
        json!({
            "file": file_name,
            "bytes": bytes,
            "sha256": sha256,
        }),
    );
    metadata.insert("vocab_size".to_string(), json!(model.n_vocab()));
    metadata.insert("threads".to_string(), json!(threads));
    metadata.insert(
        "n_gpu_layers".to_string(),
        json!(offloaded_layers(&model, scoring)),
    );
    metadata.insert("max_prompt_tokens".to_string(), json!(context_tokens));
    metadata.insert("context_tokens".to_string(), json!(actual_context));
    metadata.insert(
        "decode_chunk".to_string(),
        json!(crate::engine::DECODE_CHUNK),
    );
    metadata.insert(
        "llama_cpp_2_version".to_string(),
        json!(LLAMA_CPP_2_VERSION),
    );
    metadata.insert("tokenizers_version".to_string(), json!(TOKENIZERS_VERSION));
    metadata.insert("minijinja_version".to_string(), json!(MINIJINJA_VERSION));
    Ok(Session {
        context,
        model,
        backend,
        scoring,
        tokenizer,
        metadata: Value::Object(metadata),
        quantized: info.quantized,
    })
}

fn check_revision(source: &str, revision: &str) -> Result<bool, Error> {
    let local = Path::new(source).is_dir();
    if !local && !is_commit(revision) {
        return Err(Error::new(
            "Remote sources require a pinned 40-character revision; local sources require a revision label",
        ));
    }
    if local && revision.is_empty() {
        return Err(Error::new(
            "Local sources require an explicit revision label",
        ));
    }
    Ok(local)
}

fn is_commit(revision: &str) -> bool {
    revision.len() == 40
        && revision.bytes().all(|byte| {
            byte.is_ascii_digit() || (byte.is_ascii_lowercase() && byte.is_ascii_hexdigit())
        })
}

fn resolve_threads(threads: Option<i32>) -> Result<i32, Error> {
    match threads {
        Some(value) if value >= 1 => Ok(value),
        Some(_) => Err(Error::new("threads must be a positive integer")),
        None => {
            let count = thread::available_parallelism()
                .map(|value| value.get())
                .unwrap_or(4);
            Ok(i32::try_from(count).unwrap_or(i32::MAX))
        }
    }
}

/// llama.cpp backend selected for this build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Metal,
    // Constructed only when this build selects that backend. Tests still name every variant.
    #[cfg_attr(
        not(all(target_os = "linux", feature = "vulkan", not(feature = "cpu"))),
        allow(dead_code)
    )]
    Vulkan,
    #[cfg_attr(
        not(all(target_os = "linux", feature = "cpu", not(feature = "vulkan"))),
        allow(dead_code)
    )]
    Cpu,
}

impl Backend {
    pub fn name(self) -> &'static str {
        match self {
            Backend::Metal => "metal",
            Backend::Vulkan => "vulkan",
            Backend::Cpu => "cpu",
        }
    }
}

pub fn compiled_backend_name() -> &'static str {
    compiled_backend().name()
}

pub fn compiled_backend() -> Backend {
    #[cfg(target_os = "macos")]
    {
        return Backend::Metal;
    }
    #[cfg(all(target_os = "linux", feature = "vulkan", not(feature = "cpu")))]
    {
        return Backend::Vulkan;
    }
    #[cfg(all(target_os = "linux", feature = "cpu", not(feature = "vulkan")))]
    {
        return Backend::Cpu;
    }
    #[cfg(not(any(
        target_os = "macos",
        all(target_os = "linux", feature = "vulkan", not(feature = "cpu")),
        all(target_os = "linux", feature = "cpu", not(feature = "vulkan")),
    )))]
    {
        // `lib.rs` rejects this host or this pair of features.
        unreachable!("unsupported host or backend features")
    }
}

pub fn host_backend_supported(os: &str, arch: &str, backend: Backend) -> bool {
    match backend {
        Backend::Metal => os == "macos" && arch == "aarch64",
        Backend::Vulkan | Backend::Cpu => os == "linux" && arch == "x86_64",
    }
}

fn unsupported_host(backend: Backend) -> Error {
    let message = match backend {
        Backend::Metal => "llama.cpp Metal backend requires macOS on Apple Silicon",
        Backend::Vulkan => "llama.cpp Vulkan backend requires Linux x86_64",
        Backend::Cpu => "llama.cpp CPU backend requires Linux x86_64",
    };
    Error::new(message)
}

fn require_offload(backend: &LlamaBackend, scoring: Backend) -> Result<(), Error> {
    match scoring {
        Backend::Cpu => Ok(()),
        Backend::Metal => {
            if backend.supports_gpu_offload() {
                Ok(())
            } else {
                Err(Error::new("llama.cpp Metal GPU offload is unavailable"))
            }
        }
        Backend::Vulkan => {
            let gpu = llama_cpp_2::list_llama_ggml_backend_devices()
                .iter()
                .any(|device| device.device_type == llama_cpp_2::LlamaBackendDeviceType::Gpu);
            if backend.supports_gpu_offload() && gpu {
                Ok(())
            } else {
                Err(Error::new("no usable Vulkan GPU"))
            }
        }
    }
}

fn model_params(scoring: Backend) -> LlamaModelParams {
    match scoring {
        // Default n_gpu_layers is -1, which offloads every layer.
        Backend::Metal | Backend::Vulkan => LlamaModelParams::default(),
        Backend::Cpu => LlamaModelParams::default().with_n_gpu_layers(0),
    }
}

fn offloaded_layers(model: &LlamaModel, scoring: Backend) -> u32 {
    match scoring {
        Backend::Metal | Backend::Vulkan => model.n_layer(),
        Backend::Cpu => 0,
    }
}

fn hash_file(path: &Path) -> Result<(String, u64), Error> {
    let mut file = File::open(path).map_err(|error| {
        Error::new(format!(
            "GGUF checkpoint not found: {}: {error}",
            path.display()
        ))
    })?;
    let bytes = file.metadata().map(|meta| meta.len()).unwrap_or(0);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 8 * 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| Error::new(format!("Failed to hash {}: {error}", path.display())))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok((hex::encode(hasher.finalize()), bytes))
}

fn load_tokenizer(source: &str, revision: &str, local: bool) -> Result<ReferenceTokenizer, Error> {
    if local {
        let dir = Path::new(source);
        let template = template_in_dir(dir)?;
        return ReferenceTokenizer::from_files(&dir.join("tokenizer.json"), &template);
    }
    download_tokenizer(source, revision)
}

fn template_in_dir(dir: &Path) -> Result<String, Error> {
    let jinja = dir.join("chat_template.jinja");
    if jinja.is_file() {
        return std::fs::read_to_string(&jinja)
            .map_err(|error| Error::new(format!("Failed to read {}: {error}", jinja.display())));
    }
    let config_path = dir.join("tokenizer_config.json");
    let config: Value = serde_json::from_reader(File::open(&config_path).map_err(|error| {
        Error::new(format!("Failed to read {}: {error}", config_path.display()))
    })?)
    .map_err(|error| Error::new(format!("tokenizer_config.json is not JSON: {error}")))?;
    chat_template_from_config(&config)
}

fn download_tokenizer(model_id: &str, revision: &str) -> Result<ReferenceTokenizer, Error> {
    let (owner, name) = hf_hub::split_id(model_id);
    if owner.is_empty() || name.is_empty() {
        return Err(Error::new(format!(
            "Remote model id must be owner/name, got {model_id}"
        )));
    }
    let client = hf_hub::HFClientSync::new()
        .map_err(|error| Error::new(format!("Hugging Face client failed: {error}")))?;
    let repo = client.model(owner, name);
    let offline = std::env::var_os("HF_HUB_OFFLINE").is_some();
    let fetch = |filename: &str| {
        repo.download_file()
            .filename(filename)
            .revision(revision.to_string())
            .local_files_only(offline)
            .send()
            .map_err(|error| {
                Error::new(format!(
                    "Failed to fetch {filename} for {model_id}@{revision}: {error}"
                ))
            })
    };
    let tokenizer_json = fetch("tokenizer.json")?;
    let template = match fetch("chat_template.jinja") {
        Ok(path) => std::fs::read_to_string(&path)
            .map_err(|error| Error::new(format!("Failed to read {}: {error}", path.display())))?,
        Err(_) => {
            let config_path = fetch("tokenizer_config.json")?;
            let config: Value =
                serde_json::from_reader(File::open(&config_path).map_err(|error| {
                    Error::new(format!("Failed to read {}: {error}", config_path.display()))
                })?)
                .map_err(|error| {
                    Error::new(format!("tokenizer_config.json is not JSON: {error}"))
                })?;
            chat_template_from_config(&config)?
        }
    };
    ReferenceTokenizer::from_files(&tokenizer_json, &template)
}

fn verify_vocabulary(model: &LlamaModel, tokenizer: &ReferenceTokenizer) -> Result<(), Error> {
    let row = json!({
        "id": "vocabulary-probe",
        "state": "probe evidence",
        "question": "probe criterion?",
        "options": [
            {"id": "yes", "description": "Yes."},
            {"id": "no", "description": "No."}
        ]
    });
    let messages = direct_messages(&row)?;
    let prompt = tokenizer.render(&messages)?;
    let reference = tokenizer.encode(&prompt)?;
    let gguf_ids = tokenize_with(model, &prompt)?;
    if gguf_ids != reference {
        return Err(Error::new(
            "The GGUF vocabulary disagrees with the reference tokenizer",
        ));
    }
    for letter in LETTERS.chars() {
        let letter = letter.to_string();
        let encoded = tokenizer.encode(&letter)?;
        let piece = (encoded.len() == 1)
            .then(|| piece_bytes(model, encoded[0]))
            .transpose()?;
        if piece.as_deref() != Some(letter.as_bytes()) {
            return Err(Error::new(format!(
                "Answer slot {} is not a shared single token",
                python_repr(&letter)
            )));
        }
    }
    Ok(())
}

pub(crate) fn tokenize_with(model: &LlamaModel, text: &str) -> Result<Vec<i32>, Error> {
    let tokens = model
        .str_to_token(text, AddBos::Never)
        .map_err(|_| Error::new("The GGUF tokenizer rejected the prompt text"))?;
    Ok(tokens
        .into_iter()
        .map(|token| {
            let LlamaToken(id) = token;
            id
        })
        .collect())
}

fn piece_bytes(model: &LlamaModel, token: i32) -> Result<Vec<u8>, Error> {
    model
        .token_to_piece_bytes(LlamaToken(token), 64, true, None)
        .map_err(|_| Error::new("The GGUF tokenizer cannot render a token"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_model_requires_immutable_revision() {
        let error = match load_model(
            "Qwen/Qwen3.5-4B",
            "main",
            Path::new("/nonexistent.gguf"),
            None,
            4096,
        ) {
            Err(error) => error,
            Ok(_) => panic!("an unpinned revision must be rejected"),
        };
        assert!(error.to_string().contains("pinned"));
    }

    #[test]
    fn missing_gguf_is_rejected() {
        let error = match load_model(
            "Qwen/Qwen3.5-4B",
            "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            Path::new("/tmp/semif-absent-checkpoint.gguf"),
            None,
            4096,
        ) {
            Err(error) => error,
            Ok(_) => panic!("a missing GGUF file must be rejected"),
        };
        assert!(error.to_string().contains("GGUF checkpoint not found"));
    }

    #[test]
    fn host_matches_the_compiled_backend() {
        let cases = [
            ("macos", "aarch64", Backend::Metal, true),
            ("macos", "x86_64", Backend::Metal, false),
            ("linux", "aarch64", Backend::Metal, false),
            ("linux", "x86_64", Backend::Vulkan, true),
            ("linux", "aarch64", Backend::Vulkan, false),
            ("macos", "aarch64", Backend::Vulkan, false),
            ("linux", "x86_64", Backend::Cpu, true),
            ("linux", "aarch64", Backend::Cpu, false),
            ("macos", "x86_64", Backend::Cpu, false),
            ("macos", "aarch64", Backend::Cpu, false),
        ];
        for (os, arch, backend, supported) in cases {
            assert_eq!(
                host_backend_supported(os, arch, backend),
                supported,
                "{os} {arch} {backend:?}"
            );
        }
    }

    #[test]
    fn pins_match_the_manifest() {
        let manifest = include_str!("../Cargo.toml");
        assert!(manifest.contains("version = \"=0.1.157\""));
        assert!(manifest.contains("tokenizers = \"=0.23.2\""));
        assert!(manifest.contains("version = \"=2.24.0\""));
    }

    #[test]
    fn threads_must_be_positive() {
        assert!(resolve_threads(Some(0))
            .unwrap_err()
            .to_string()
            .contains("positive"));
        assert_eq!(resolve_threads(Some(4)).unwrap(), 4);
    }
}
