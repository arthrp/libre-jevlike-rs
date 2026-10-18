mod cli;

fn main() -> std::process::ExitCode {
    cli::run_from(std::env::args_os())
}
