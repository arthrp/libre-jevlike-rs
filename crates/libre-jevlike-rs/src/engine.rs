//! Direct-mode llama.cpp decode.
//!
//! The context is cleared, then the prompt is decoded in 512-token chunks.
//! Only the final token requests logits. An empty token list is refused before
//! any native call.

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::token::LlamaToken;

use crate::error::Error;

pub const DECODE_CHUNK: usize = 512;

/// Metadata records llama.cpp's actual context length, which can exceed the request.
pub fn recorded_context_tokens(actual_n_ctx: u32) -> u32 {
    actual_n_ctx
}

pub fn refuse_empty_decode(tokens: &[i32]) -> Result<(), Error> {
    if tokens.is_empty() {
        Err(Error::new("Refusing to decode an empty token list"))
    } else {
        Ok(())
    }
}

pub fn full_logits(context: &mut LlamaContext<'_>, tokens: &[i32]) -> Result<Vec<f32>, Error> {
    refuse_empty_decode(tokens)?;
    context.clear_kv_cache();
    decode(context, tokens, 0, true)?;
    let logits = context.get_logits();
    if logits.is_empty() {
        return Err(Error::new(
            "llama.cpp returned no logits for the flagged position",
        ));
    }
    Ok(logits.to_vec())
}

fn decode(
    context: &mut LlamaContext<'_>,
    tokens: &[i32],
    start: i32,
    want_logits: bool,
) -> Result<(), Error> {
    refuse_empty_decode(tokens)?;
    let chunk_limit = DECODE_CHUNK.min(usize::try_from(context.n_batch()).unwrap_or(DECODE_CHUNK));
    let chunk_limit = chunk_limit.max(1);
    let total = tokens.len();
    let mut offset = 0;
    while offset < total {
        let end = (offset + chunk_limit).min(total);
        let chunk = &tokens[offset..end];
        let mut batch = LlamaBatch::new(chunk.len(), 1);
        for (index, token) in chunk.iter().copied().enumerate() {
            let position = start
                + i32::try_from(offset + index)
                    .map_err(|_| Error::new("Token position does not fit in i32"))?;
            let logits = want_logits && offset + index + 1 == total;
            batch
                .add(LlamaToken(token), position, &[0], logits)
                .map_err(|error| {
                    Error::new(format!("Failed to build the decode batch: {error}"))
                })?;
        }
        context
            .decode(&mut batch)
            .map_err(|_| Error::new("llama_decode failed; raise --max-tokens if prompts grew"))?;
        offset = end;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_refuses_empty_decode() {
        let error = refuse_empty_decode(&[]).unwrap_err();
        assert!(error.to_string().contains("empty"));
    }

    #[test]
    fn context_tokens_record_the_native_size() {
        // llama.cpp may round the requested window. The Python test stubs
        // llama_n_ctx to 4352 after a request of 4160; metadata stores that value.
        assert_eq!(recorded_context_tokens(4352), 4352);
        assert_ne!(4160, recorded_context_tokens(4352));
    }
}
