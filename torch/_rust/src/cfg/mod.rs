// CPython 3.12 bytecode -> SSA control-flow graph.
//
// Pipeline: decode (bytecode.rs) -> block partition (blocks.rs) -> abstract
// interpretation (interp.rs) -> optional simplify (simplify.rs) -> optional
// outlining (outline.rs) -> verify (verify.rs). The IR is in ir.rs; python.rs
// is the PyO3 surface.

pub mod blocks;
pub mod bytecode;
pub mod guard;
pub mod inline;
pub mod interp;
pub mod ir;
pub mod opcodes;
pub mod optimize;
pub mod outline;
pub mod partition;
pub mod python;
pub mod simplify;
pub mod specialize;
pub mod verify;

use std::fmt;

#[derive(Debug)]
pub enum CfgError {
    // Input the converter doesn't handle (surfaced as NotImplementedError).
    Unsupported {
        offset: u32,
        opname: String,
        reason: String,
    },
    // Internal bug caught by the verifier (surfaced as RuntimeError).
    Verify(String),
}

impl CfgError {
    pub fn unsupported(offset: u32, opname: &str, reason: impl Into<String>) -> Self {
        CfgError::Unsupported {
            offset,
            opname: opname.to_string(),
            reason: reason.into(),
        }
    }
}

impl fmt::Display for CfgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CfgError::Unsupported {
                offset,
                opname,
                reason,
            } => write!(f, "offset {offset} {opname}: {reason}"),
            CfgError::Verify(msg) => write!(f, "CFG verification failed: {msg}"),
        }
    }
}

pub fn build(info: bytecode::CodeInfo, simplify: bool, outline: Option<outline::Outline>) -> Result<ir::Cfg, CfgError> {
    let dec = bytecode::decode(&info)?;
    let part = blocks::partition(&dec)?;
    let cfg = ir::Cfg::new(info.slot_names, info.const_reprs, info.names);
    let mut cfg = interp::interpret(cfg, &dec, &part)?;
    if cfg!(debug_assertions) || !simplify {
        verify::verify(&cfg).map_err(CfgError::Verify)?;
    }
    if simplify {
        simplify::simplify(&mut cfg);
        verify::verify(&cfg).map_err(CfgError::Verify)?;
    }
    if let Some(mode) = outline {
        outline::outline_chains(&mut cfg, mode);
        verify::verify(&cfg).map_err(CfgError::Verify)?;
    }
    Ok(cfg)
}
