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

mod cli;
mod engine;
mod error;
mod gguf;
mod loader;
mod prompt;
mod row;
mod score;

pub use cli::run_from;
pub use error::Error;
pub use loader::{compiled_backend_name, load_model, Session};
