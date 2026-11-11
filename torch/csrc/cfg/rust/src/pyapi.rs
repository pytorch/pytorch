//! The Python face of the crate.

use crate::bytecode::{self, OpcodeTable};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// The instructions the frontend sees in a code object, as (offset, opcode
/// name, argument, jump target) tuples.
#[pyfunction]
pub fn decode(
    py: Python<'_>,
    code: &Bound<'_, PyAny>,
) -> PyResult<Vec<(u32, String, u32, Option<u32>)>> {
    let table = OpcodeTable::load(py)?;
    let co_code = code.getattr("co_code")?;
    let co_code = co_code.downcast::<PyBytes>()?;
    Ok(bytecode::decode(&table, co_code)
        .into_iter()
        .map(|ins| (ins.off, table.names[ins.code as usize].clone(), ins.arg, ins.target))
        .collect())
}
