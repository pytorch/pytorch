//! The compiler and virtual machine behind `torch.vm`.
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
mod inline;
mod ir;
mod pyapi;
mod pyffi;
mod shapes;
mod shortcuts;
mod spec;
mod unroll;
mod vm;

use pyo3::prelude::*;

// The VM's state is only guarded by the GIL.
#[pymodule(gil_used = true)]
fn _cfg_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pyapi::init, m)?)?;
    m.add_function(wrap_pyfunction!(pyapi::compile, m)?)?;
    m.add_function(wrap_pyfunction!(pyapi::decode, m)?)?;
    m.add_function(wrap_pyfunction!(pyapi::builtin_passes, m)?)?;
    m.add_function(wrap_pyfunction!(pyapi::register_call_shortcut, m)?)?;
    m.add_function(wrap_pyfunction!(pyapi::register_getattr_shortcut, m)?)?;
    m.add_function(wrap_pyfunction!(pyapi::clear_shortcuts, m)?)?;
    m.add_function(wrap_pyfunction!(pyapi::profiling_depth, m)?)?;
    m.add_function(wrap_pyfunction!(pyapi::frames_offered, m)?)?;
    m.add_function(wrap_pyfunction!(pyapi::leave_to_interpreter, m)?)?;
    m.add_function(wrap_pyfunction!(pyapi::load_passes, m)?)?;
    m.add_class::<pyapi::NativePass>()?;
    m.add_function(wrap_pyfunction!(pyapi::plan_changed, m)?)?;
    m.add_function(wrap_pyfunction!(pyapi::unsettled, m)?)?;
    m.add_function(wrap_pyfunction!(pyapi::module_stamp, m)?)?;
    m.add_function(wrap_pyfunction!(pyapi::module_identity, m)?)?;
    m.add_function(wrap_pyfunction!(shapes::tensor_key, m)?)?;
    m.add_function(wrap_pyfunction!(shapes::tensor_stamp, m)?)?;
    m.add_function(wrap_pyfunction!(shapes::snapshot, m)?)?;
    m.add_class::<shapes::Snapshot>()?;
    m.add_class::<pyapi::CfgFunction>()?;
    m.add_class::<pyapi::Region>()?;
    m.add_class::<pyapi::Replay>()?;
    m.add_class::<shapes::ShapeGuard>()?;
    // What a leap is called with in place of a value that is not defined.
    m.add("UNDEFINED", unsafe { Bound::from_borrowed_ptr(m.py(), vm::undefined()) })?;
    Ok(())
}
