//! The compiler and virtual machine behind `torch.cfg_compile`.
//!
//! A Python function is turned into a control-flow graph in SSA form
//! (`frontend`), which a register VM executes (`vm`). Blocks start out in a
//! generic tier with the semantics of the bytecode, are profiled on real
//! tensors with the dispatcher's capture hook (`capi`), and are then
//! specialized by a pipeline of passes (`spec`).

mod analysis;
mod bytecode;
mod capi;
mod frontend;
mod ir;
mod pyapi;
mod pyffi;
mod spec;
mod vm;

use pyo3::prelude::*;

#[pymodule]
fn _cfg_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pyapi::init, m)?)?;
    m.add_function(wrap_pyfunction!(pyapi::compile, m)?)?;
    m.add_function(wrap_pyfunction!(pyapi::builtin_passes, m)?)?;
    m.add_class::<pyapi::CfgFunction>()?;
    Ok(())
}
