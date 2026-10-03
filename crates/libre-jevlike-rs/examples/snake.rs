//! Play snake-app over TCP by scoring the next direction.
//!
//! Start the game with `npm start -- --tcp --step`, then run this example.
//! Each state line becomes one decision row. The highest-probability option
//! is sent back as `l`, `r`, `u`, or `d`.

use std::collections::HashSet;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use serde::Deserialize;
use serde_json::{json, Value};

/// Score snake moves against a TCP game server.
#[derive(Parser, Debug)]
#[command(name = "snake", about = "Play snake-app by scoring direction options")]
struct Args {
    /// Hugging Face model id or a local directory with the tokenizer and chat template.
    #[arg(long)]
    model: String,
    /// 40-character commit for a remote model, or a revision label for a local directory.
    #[arg(long)]
    revision: String,
    /// Local GGUF checkpoint.
    #[arg(long)]
    gguf: PathBuf,
    /// snake-app control address.
    #[arg(long, default_value = "127.0.0.1:5555")]
    addr: String,
    /// Stop after this many finished games.
    #[arg(long, default_value_t = 1)]
    games: u32,
    /// Maximum prompt tokens. Prompts are never truncated.
    #[arg(long, default_value_t = 4096)]
    max_tokens: u32,
    /// CPU threads for llama.cpp. Defaults to the visible core count.
    #[arg(long)]
    llama_threads: Option<i32>,
}

#[derive(Debug, Deserialize)]
struct StateMessage {
    #[serde(rename = "type")]
    kind: String,
    status: String,
    width: i32,
    height: i32,
    snake: Vec<Cell>,
    direction: String,
    food: Option<Cell>,
    score: i64,
}

#[derive(Debug, Clone, Copy, Deserialize)]
struct Cell {
    x: i32,
    y: i32,
}

struct Move {
    id: &'static str,
    name: &'static str,
    dx: i32,
    dy: i32,
}

const MOVES: [Move; 4] = [
    Move {
        id: "l",
        name: "left",
        dx: -1,
        dy: 0,
    },
    Move {
        id: "r",
        name: "right",
        dx: 1,
        dy: 0,
    },
    Move {
        id: "u",
        name: "up",
        dx: 0,
        dy: -1,
    },
    Move {
        id: "d",
        name: "down",
        dx: 0,
        dy: 1,
    },
];

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(1)
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.games == 0 {
        return Err("games must be a positive integer".into());
    }

    libre_jevlike_rs::silence_llama_logs();
    let mut session = libre_jevlike_rs::load_model(
        &args.model,
        &args.revision,
        &args.gguf,
        args.llama_threads,
        args.max_tokens,
    )?;

    eprintln!("connecting to {}", args.addr);
    let stream = TcpStream::connect(&args.addr)
        .map_err(|error| format!("could not connect to {}: {error}", args.addr))?;
    stream.set_nodelay(true)?;
    let mut reader = BufReader::new(stream);

    let mut finished = 0u32;
    let mut step = 0u32;
    loop {
        let state = read_state(&mut reader)?;
        match state.status.as_str() {
            "running" => {
                step += 1;
                let row = decision_row(&state, step)?;
                let result = session.score(&row, args.max_tokens)?;
                let (direction, summary) = chosen_direction(&result)?;
                println!("step {step} {direction} {summary}");
                send(&mut reader, &direction)?;
            }
            "waiting" | "over" | "won" => {
                println!("{} score {}", state.status, state.score);
                if state.status != "waiting" {
                    finished += 1;
                }
                if finished >= args.games {
                    break;
                }
                step = 0;
                send(&mut reader, "r")?;
            }
            other => return Err(format!("unknown status {other}").into()),
        }
    }
    Ok(())
}

fn read_state(
    reader: &mut BufReader<TcpStream>,
) -> Result<StateMessage, Box<dyn std::error::Error>> {
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Err("connection closed".into());
        }
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed == "ok" {
            continue;
        }
        if let Some(message) = trimmed.strip_prefix("error:") {
            return Err(format!("server error:{message}").into());
        }
        if !trimmed.starts_with('{') {
            return Err(format!("unexpected message: {trimmed}").into());
        }
        let state: StateMessage = serde_json::from_str(trimmed)?;
        if state.kind != "state" {
            return Err(format!("unexpected message: {trimmed}").into());
        }
        return Ok(state);
    }
}

fn send(reader: &mut BufReader<TcpStream>, command: &str) -> Result<(), std::io::Error> {
    let stream = reader.get_mut();
    writeln!(stream, "{command}")?;
    stream.flush()
}

fn decision_row(state: &StateMessage, step: u32) -> Result<Value, Box<dyn std::error::Error>> {
    if state.snake.is_empty() {
        return Err("running state has no snake".into());
    }
    let head = state.snake[0];
    let blocked = blocked_cells(&state.snake);
    let reverse = opposite(&state.direction)
        .ok_or_else(|| format!("unknown direction {}", state.direction))?;
    let mut options = Vec::new();
    for mv in &MOVES {
        if mv.id == reverse {
            continue;
        }
        options.push(json!({
            "id": mv.id,
            "description": describe_move(mv, head, state, &blocked),
        }));
    }

    Ok(json!({
        "id": format!("step-{step}"),
        "state": {
            "board": board(state)?,
            "head": {"x": head.x, "y": head.y},
            "direction": direction_name(&state.direction)?,
            "food": match state.food {
                Some(cell) => json!({"x": cell.x, "y": cell.y}),
                None => Value::Null,
            },
            "length": state.snake.len(),
            "score": state.score,
        },
        "question": "Which move keeps the snake alive and brings its head closer to the food?",
        "options": options,
    }))
}

fn board(state: &StateMessage) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let width = usize::try_from(state.width).map_err(|_| "board size must be positive")?;
    let height = usize::try_from(state.height).map_err(|_| "board size must be positive")?;
    if width == 0 || height == 0 {
        return Err("board size must be positive".into());
    }
    let mut rows = vec![vec!['.'; width]; height];
    if let Some(food) = state.food {
        paint(&mut rows, food, 'F')?;
    }
    for (index, cell) in state.snake.iter().enumerate() {
        paint(&mut rows, *cell, if index == 0 { 'H' } else { 'o' })?;
    }
    Ok(rows
        .into_iter()
        .map(|row| row.into_iter().collect())
        .collect())
}

fn paint(rows: &mut [Vec<char>], cell: Cell, mark: char) -> Result<(), Box<dyn std::error::Error>> {
    let y = usize::try_from(cell.y)
        .map_err(|_| format!("cell ({},{}) is off the board", cell.x, cell.y))?;
    let x = usize::try_from(cell.x)
        .map_err(|_| format!("cell ({},{}) is off the board", cell.x, cell.y))?;
    let row = rows
        .get_mut(y)
        .ok_or_else(|| format!("cell ({},{}) is off the board", cell.x, cell.y))?;
    let slot = row
        .get_mut(x)
        .ok_or_else(|| format!("cell ({},{}) is off the board", cell.x, cell.y))?;
    *slot = mark;
    Ok(())
}

/// Every occupied cell except the tail. The tail moves away this tick unless the
/// snake grows, and growth only happens on the food cell, which is never the tail.
fn blocked_cells(snake: &[Cell]) -> HashSet<(i32, i32)> {
    snake
        .iter()
        .take(snake.len().saturating_sub(1))
        .map(|cell| (cell.x, cell.y))
        .collect()
}

fn describe_move(
    mv: &Move,
    head: Cell,
    state: &StateMessage,
    blocked: &HashSet<(i32, i32)>,
) -> String {
    let x = head.x + mv.dx;
    let y = head.y + mv.dy;
    if x < 0 || y < 0 || x >= state.width || y >= state.height {
        return format!("Move {} to ({x},{y}): hits the wall.", mv.name);
    }
    if blocked.contains(&(x, y)) {
        return format!("Move {} to ({x},{y}): hits the body.", mv.name);
    }
    match state.food {
        Some(food) => {
            let before = (head.x - food.x).abs() + (head.y - food.y).abs();
            let after = (x - food.x).abs() + (y - food.y).abs();
            format!(
                "Move {} to ({x},{y}): free; food distance {before} -> {after}.",
                mv.name
            )
        }
        None => format!("Move {} to ({x},{y}): free.", mv.name),
    }
}

fn opposite(direction: &str) -> Option<&'static str> {
    match direction {
        "l" => Some("r"),
        "r" => Some("l"),
        "u" => Some("d"),
        "d" => Some("u"),
        _ => None,
    }
}

fn direction_name(direction: &str) -> Result<&'static str, String> {
    match direction {
        "l" => Ok("left"),
        "r" => Ok("right"),
        "u" => Ok("up"),
        "d" => Ok("down"),
        _ => Err(format!("unknown direction {direction}")),
    }
}

fn chosen_direction(result: &Value) -> Result<(String, String), Box<dyn std::error::Error>> {
    let ids = result["option_ids"]
        .as_array()
        .ok_or("score is missing option_ids")?;
    let probabilities = result["probabilities"]
        .as_array()
        .ok_or("score is missing probabilities")?;
    if ids.is_empty() || ids.len() != probabilities.len() {
        return Err("score options and probabilities disagree".into());
    }
    let mut chosen = String::new();
    let mut best = f64::NEG_INFINITY;
    let mut parts = Vec::with_capacity(ids.len());
    for (id, probability) in ids.iter().zip(probabilities) {
        let id = id.as_str().ok_or("option id is not a string")?;
        let probability = probability.as_f64().ok_or("probability is not a number")?;
        if probability > best {
            best = probability;
            chosen = id.to_string();
        }
        parts.push(format!("{id}={probability:.3}"));
    }
    if chosen.is_empty() {
        return Err("score did not select an option".into());
    }
    Ok((chosen, parts.join(" ")))
}
