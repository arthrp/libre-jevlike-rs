//! Minimal GGUF metadata reader.
//!
//! Only `general.architecture` and `general.file_type` are required. The rest of
//! the metadata is skipped so tensor payloads are never read.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::error::Error;

/// Architecture names from llama.cpp `LLM_ARCH_NAMES` for dense Qwen3 and Qwen3.5.
pub const SUPPORTED_ARCHITECTURES: &[&str] = &["qwen3", "qwen35"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufInfo {
    pub architecture: String,
    pub file_type: u32,
    pub file_type_name: String,
    pub quantized: bool,
}

pub fn read_info(path: &Path) -> Result<GgufInfo, Error> {
    let mut file = File::open(path).map_err(|error| {
        Error::new(format!(
            "GGUF checkpoint not found: {}: {error}",
            path.display()
        ))
    })?;
    let mut magic = [0u8; 4];
    read_exact(&mut file, &mut magic)?;
    if &magic != b"GGUF" {
        return Err(Error::new(format!(
            "GGUF checkpoint is not a GGUF file: {}",
            path.display()
        )));
    }
    let version = read_u32(&mut file)?;
    if version < 2 {
        return Err(Error::new(format!(
            "GGUF version {version} is not supported"
        )));
    }
    let _tensor_count = read_u64(&mut file)?;
    let metadata_count = read_u64(&mut file)?;
    let mut architecture = None;
    let mut file_type = None;
    for _ in 0..metadata_count {
        let key = read_string(&mut file)?;
        let value_type = read_u32(&mut file)?;
        if key == "general.architecture" && value_type == GGUF_STRING {
            architecture = Some(read_string(&mut file)?);
        } else if key == "general.file_type" && matches!(value_type, GGUF_UINT32 | GGUF_INT32) {
            file_type = Some(read_u32(&mut file)?);
        } else {
            skip_value(&mut file, value_type)?;
        }
        if architecture.is_some() && file_type.is_some() {
            break;
        }
    }
    let architecture = architecture.ok_or_else(|| {
        Error::new(format!(
            "GGUF metadata is missing general.architecture: {}",
            path.display()
        ))
    })?;
    if !SUPPORTED_ARCHITECTURES.contains(&architecture.as_str()) {
        return Err(Error::new(format!(
            "GGUF architecture '{architecture}' is not supported; llama.cpp names for this build are qwen3 and qwen35"
        )));
    }
    let file_type = file_type.ok_or_else(|| {
        Error::new(format!(
            "GGUF metadata is missing general.file_type: {}",
            path.display()
        ))
    })?;
    let (file_type_name, quantized) = file_type_name(file_type);
    Ok(GgufInfo {
        architecture,
        file_type,
        file_type_name,
        quantized,
    })
}

const GGUF_UINT8: u32 = 0;
const GGUF_INT8: u32 = 1;
const GGUF_UINT16: u32 = 2;
const GGUF_INT16: u32 = 3;
const GGUF_UINT32: u32 = 4;
const GGUF_INT32: u32 = 5;
const GGUF_FLOAT32: u32 = 6;
const GGUF_BOOL: u32 = 7;
const GGUF_STRING: u32 = 8;
const GGUF_ARRAY: u32 = 9;
const GGUF_UINT64: u32 = 10;
const GGUF_INT64: u32 = 11;
const GGUF_FLOAT64: u32 = 12;

fn file_type_name(file_type: u32) -> (String, bool) {
    let name = match file_type {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        31 => "IQ1_M",
        32 => "BF16",
        36 => "TQ1_0",
        37 => "TQ2_0",
        38 => "MXFP4_MOE",
        39 => "NVFP4",
        40 => "Q1_0",
        41 => "Q2_0",
        other => {
            return (
                format!("unknown-{other}"),
                other != 0 && other != 1 && other != 32,
            )
        }
    };
    let quantized = !matches!(file_type, 0 | 1 | 32);
    (name.to_string(), quantized)
}

pub fn dtype_name(info: &GgufInfo) -> String {
    format!("gguf:{}", info.file_type_name)
}

fn read_exact(file: &mut File, buf: &mut [u8]) -> Result<(), Error> {
    file.read_exact(buf)
        .map_err(|error| Error::new(format!("GGUF metadata is truncated: {error}")))
}

fn read_u32(file: &mut File) -> Result<u32, Error> {
    let mut buf = [0u8; 4];
    read_exact(file, &mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(file: &mut File) -> Result<u64, Error> {
    let mut buf = [0u8; 8];
    read_exact(file, &mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_string(file: &mut File) -> Result<String, Error> {
    let len =
        usize::try_from(read_u64(file)?).map_err(|_| Error::new("GGUF string is too large"))?;
    let mut buf = vec![0u8; len];
    read_exact(file, &mut buf)?;
    String::from_utf8(buf).map_err(|error| Error::new(format!("GGUF string is not UTF-8: {error}")))
}

fn skip_value(file: &mut File, value_type: u32) -> Result<(), Error> {
    let bytes = match value_type {
        GGUF_UINT8 | GGUF_INT8 | GGUF_BOOL => 1,
        GGUF_UINT16 | GGUF_INT16 => 2,
        GGUF_UINT32 | GGUF_INT32 | GGUF_FLOAT32 => 4,
        GGUF_UINT64 | GGUF_INT64 | GGUF_FLOAT64 => 8,
        GGUF_STRING => {
            let len = read_u64(file)?;
            file.seek(SeekFrom::Current(
                i64::try_from(len).map_err(|_| Error::new("GGUF string is too large"))?,
            ))
            .map_err(|error| Error::new(format!("GGUF metadata is truncated: {error}")))?;
            return Ok(());
        }
        GGUF_ARRAY => {
            let element = read_u32(file)?;
            let count = read_u64(file)?;
            for _ in 0..count {
                skip_value(file, element)?;
            }
            return Ok(());
        }
        other => {
            return Err(Error::new(format!(
                "GGUF metadata value type {other} is not supported"
            )))
        }
    };
    file.seek(SeekFrom::Current(i64::from(bytes)))
        .map_err(|error| Error::new(format!("GGUF metadata is truncated: {error}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn gguf_with(architecture: &str, file_type: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend(b"GGUF");
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
        bytes.extend(3u64.to_le_bytes());
        push_string(&mut bytes, "general.alignment");
        bytes.extend(GGUF_UINT32.to_le_bytes());
        bytes.extend(32u32.to_le_bytes());
        push_string(&mut bytes, "general.architecture");
        bytes.extend(GGUF_STRING.to_le_bytes());
        push_string(&mut bytes, architecture);
        push_string(&mut bytes, "general.file_type");
        bytes.extend(GGUF_UINT32.to_le_bytes());
        bytes.extend(file_type.to_le_bytes());
        bytes
    }

    fn push_string(bytes: &mut Vec<u8>, text: &str) {
        bytes.extend((text.len() as u64).to_le_bytes());
        bytes.extend(text.as_bytes());
    }

    #[test]
    fn reads_qwen_architecture_and_quant_name() {
        let dir = std::env::temp_dir().join(format!("semif-gguf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&gguf_with("qwen35", 15))
            .unwrap();
        let info = read_info(&path).unwrap();
        assert_eq!(info.architecture, "qwen35");
        assert_eq!(info.file_type_name, "Q4_K_M");
        assert!(info.quantized);
        assert_eq!(dtype_name(&info), "gguf:Q4_K_M");
        let bf16 = dir.join("bf16.gguf");
        std::fs::File::create(&bf16)
            .unwrap()
            .write_all(&gguf_with("qwen3", 32))
            .unwrap();
        let info = read_info(&bf16).unwrap();
        assert!(!info.quantized);
        assert_eq!(dtype_name(&info), "gguf:BF16");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_other_architectures_and_bad_magic() {
        let dir = std::env::temp_dir().join(format!("semif-gguf-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("other.gguf");
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&gguf_with("llama", 1))
            .unwrap();
        assert!(read_info(&path)
            .unwrap_err()
            .to_string()
            .contains("not supported"));
        let bad = dir.join("bad.gguf");
        std::fs::write(&bad, b"not-a-gguf").unwrap();
        assert!(read_info(&bad)
            .unwrap_err()
            .to_string()
            .contains("not a GGUF"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
