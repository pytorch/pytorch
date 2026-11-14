//! The compiler behind `torch.cfg_compile`.
//!
//! So far: the IR a Python function is compiled to (`ir`), a control-flow
//! graph in SSA form, and the decoder for the bytecode it is built from
//! (`bytecode`).

// Nothing constructs or runs the IR yet; the commits above add the frontend
// and the VM.
#![allow(dead_code)]

mod bytecode;
mod ir;
mod pyapi;
mod pyffi;

use pyo3::prelude::*;

#[pymodule]
fn _cfg_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pyapi::decode, m)?)?;
    Ok(())
}
