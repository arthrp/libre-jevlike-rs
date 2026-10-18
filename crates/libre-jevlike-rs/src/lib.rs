//! Direct option-logit scoring through llama.cpp.

#[cfg(not(any(target_os = "macos", all(target_os = "linux", target_arch = "x86_64"),)))]
compile_error!("supported hosts are macOS and Linux x86_64");

#[cfg(all(
    target_os = "linux",
    target_arch = "x86_64",
    feature = "vulkan",
    feature = "cpu",
))]
compile_error!("enable exactly one of the vulkan and cpu features");

#[cfg(all(
    target_os = "linux",
    target_arch = "x86_64",
    not(feature = "vulkan"),
    not(feature = "cpu"),
))]
compile_error!("enable exactly one of the vulkan and cpu features");

mod engine;
mod error;
mod gguf;
mod loader;
mod prompt;
mod row;
mod score;

pub use error::Error;
pub use loader::{compiled_backend_name, load_model, Session};
pub use row::validate_row;

/// Silence llama.cpp's own logs so a caller can keep stdout limited to scores.
pub fn silence_llama_logs() {
    llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default().with_logs_enabled(false));
}
