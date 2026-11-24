//! The compiler and virtual machine behind `torch.vm`.
//!
//! A Python function is turned into a control-flow graph in SSA form
//! (`frontend`), which a register VM executes (`vm`). Blocks start out in a
//! generic tier with the semantics of the bytecode, are profiled on real
//! tensors with the dispatcher's capture hook (`capi`), and are then
//! specialized by a pipeline of passes (`spec`). The tensor work of a
//! specialized block is described as graphs (`graph`).

mod analysis;
mod asks;
mod backward;
mod bytecode;
mod capi;
mod chunks;
mod frontend;
mod graph;
mod inline;
mod ir;
mod pyapi;
mod pyffi;
mod shapes;
mod shortcuts;
mod spec;
mod joint;
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
    m.add_function(wrap_pyfunction!(pyapi::trial_settles, m)?)?;
    m.add_function(wrap_pyfunction!(pyapi::interrupted, m)?)?;
    m.add_function(wrap_pyfunction!(pyapi::said, m)?)?;
    m.add_function(wrap_pyfunction!(shapes::tensor_key, m)?)?;
    m.add_class::<pyapi::CfgFunction>()?;
    m.add_class::<pyapi::Region>()?;
    m.add_class::<pyapi::Replay>()?;
    m.add_class::<shapes::ShapeGuard>()?;
    m.add_class::<joint::Joint>()?;
    m.add_class::<backward::Staged>()?;
    m.add_function(wrap_pyfunction!(backward::backward_forms, m)?)?;
    m.add_function(wrap_pyfunction!(backward::backward_served, m)?)?;
    Ok(())
}
