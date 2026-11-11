//! The compiler and virtual machine behind `torch.cfg_compile`.
//!
//! A Python function is turned into a control-flow graph in SSA form
//! (`frontend`), which a register VM executes (`vm`).

// Parts of the analyses are only read by the tiers that later commits add.
#![allow(dead_code)]

mod analysis;
mod bytecode;
mod frontend;
mod ir;
mod pyapi;
mod pyffi;
mod vm;

use pyo3::prelude::*;

#[pymodule]
fn _cfg_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pyapi::compile, m)?)?;
    m.add_class::<pyapi::CfgFunction>()?;
    Ok(())
}
