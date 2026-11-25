// PyO3 surface for the bytecode -> CFG converter.

use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use super::bytecode::CodeInfo;
use super::outline::Outline;
use crate::exec::ffi::tr_code_localsplusnames;
use super::{build, opcodes, CfgError};

fn strings(code: &Bound<'_, PyAny>, attr: &str) -> PyResult<Vec<String>> {
    code.getattr(attr)?.extract()
}

fn bytes(code: &Bound<'_, PyAny>, attr: &str) -> PyResult<Vec<u8>> {
    Ok(code.getattr(attr)?.cast::<PyBytes>()?.as_bytes().to_vec())
}

pub fn code_info(code: &Bound<'_, PyAny>) -> PyResult<CodeInfo> {
    if !code.is_instance_of::<pyo3::types::PyCode>() {
        return Err(pyo3::exceptions::PyTypeError::new_err("expected a code object"));
    }
    let mut const_reprs = vec![];
    for c in code.getattr("co_consts")?.try_iter()? {
        let c = c?;
        // Code-object reprs contain addresses; keep the output deterministic.
        const_reprs.push(if c.hasattr("co_code")? {
            format!("<code {}>", c.getattr("co_qualname")?)
        } else {
            c.repr()?.to_string()
        });
    }
    Ok(CodeInfo {
        code: bytes(code, "co_code")?,
        exception_table: bytes(code, "co_exceptiontable")?,
        const_reprs,
        names: strings(code, "co_names")?,
        slot_names: unsafe {
            let names = tr_code_localsplusnames(code.as_ptr());
            Bound::from_borrowed_ptr(code.py(), names).extract()?
        },
    })
}

// None, "merge" or "origin" (see outline.rs).
pub fn outline_mode(mode: Option<&str>) -> PyResult<Option<Outline>> {
    match mode {
        None => Ok(None),
        Some("merge") => Ok(Some(Outline::Merge)),
        Some("origin") => Ok(Some(Outline::OriginOnly)),
        Some(m) => Err(pyo3::exceptions::PyValueError::new_err(format!("unknown outline mode {m:?}"))),
    }
}

pub fn to_pyerr(e: CfgError) -> PyErr {
    match e {
        CfgError::Unsupported { .. } => PyNotImplementedError::new_err(e.to_string()),
        CfgError::Verify(_) => PyRuntimeError::new_err(e.to_string()),
    }
}

/// Convert a code object (or a function, via `__code__`) into an SSA CFG and
/// return its text form. `outline` ("merge" or "origin") outlines chains of
/// blocks (see outline.rs).
#[pyfunction]
#[pyo3(signature = (code, simplify=true, raw_ids=false, origins=false, outline=None))]
pub fn bytecode_to_cfg(
    code: &Bound<'_, PyAny>,
    simplify: bool,
    raw_ids: bool,
    origins: bool,
    outline: Option<&str>,
) -> PyResult<String> {
    let code = if code.hasattr("__code__")? {
        code.getattr("__code__")?
    } else {
        code.clone()
    };
    let info = code_info(&code)?;
    let cfg = build(info, simplify, outline_mode(outline)?).map_err(to_pyerr)?;
    Ok(cfg.render(raw_ids, origins))
}

/// (name, number) for every opcode in the 3.12 table (for tests).
#[pyfunction]
pub fn _cfg_opcodes() -> Vec<(String, u8)> {
    opcodes::TABLE.iter().map(|(n, op, _)| (n.to_string(), *op)).collect()
}

/// Stack signature of an opcode as (inputs used?, outputs) where an output is
/// the index of the kept input or -1 for a new value (for tests).
#[pyfunction]
pub fn _cfg_signature(opcode: u8, arg: u32, jump: bool) -> Option<(Vec<bool>, Vec<i64>)> {
    opcodes::signature(opcode, arg, jump).map(|s| {
        (
            s.inputs.iter().map(|i| *i == opcodes::In::Used).collect(),
            s.outputs
                .iter()
                .map(|o| match o {
                    opcodes::Out::Keep(j) => *j as i64,
                    opcodes::Out::New => -1,
                })
                .collect(),
        )
    })
}
