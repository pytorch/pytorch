//! The compiler behind `torch.cfg_compile`.
//!
//! A Python function is turned into a control-flow graph in SSA form
//! (`frontend`), in an IR whose instructions have the semantics of the
//! bytecode (`ir`).

// The frontend already records what the VM of the next commit needs to run
// the graph (kill lists, the calling convention of the function).
#![allow(dead_code)]

mod bytecode;
mod frontend;
mod ir;
mod pyapi;
mod pyffi;

use pyo3::prelude::*;

#[pymodule]
fn _cfg_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pyapi::compile, m)?)?;
    m.add_class::<pyapi::CfgFunction>()?;
    Ok(())
}
