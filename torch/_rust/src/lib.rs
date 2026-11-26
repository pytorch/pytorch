use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet, PyList, PySet, PyTuple};

mod bindings;
mod bytecode_printer;
mod cfg;
mod exec;

use bindings::Tensor;

#[pyfunction]
fn test_function() -> &'static str {
    "hello from rust"
}

#[pyfunction]
fn tensor_size(t: &Tensor) -> Vec<i64> {
    t.sizes().to_vec()
}

#[pyfunction]
fn collect_tensors<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, Tensor>>> {
    let mut out = Vec::new();
    collect_into(obj, &mut out)?;
    Ok(out)
}

fn collect_into<'py>(
    obj: &Bound<'py, PyAny>,
    out: &mut Vec<Bound<'py, Tensor>>,
) -> PyResult<()> {
    if let Ok(t) = obj.cast::<Tensor>() {
        out.push(t.clone());
        return Ok(());
    }
    if let Ok(d) = obj.cast::<PyDict>() {
        for (_, v) in d.iter() {
            collect_into(&v, out)?;
        }
    } else if let Ok(seq) = obj.cast::<PyList>() {
        for item in seq.iter() {
            collect_into(&item, out)?;
        }
    } else if let Ok(seq) = obj.cast::<PyTuple>() {
        for item in seq.iter() {
            collect_into(&item, out)?;
        }
    } else if let Ok(s) = obj.cast::<PySet>() {
        for item in s.iter() {
            collect_into(&item, out)?;
        }
    } else if let Ok(s) = obj.cast::<PyFrozenSet>() {
        for item in s.iter() {
            collect_into(&item, out)?;
        }
    }
    Ok(())
}

// Declare the module safe to run without the GIL so the free-threaded CPython
// build (e.g. py3.14t) does not re-enable the GIL and emit a RuntimeWarning when
// torch._rust is imported. Any shared mutable state must be thread-safe without
// relying on the GIL (see bytecode_printer.rs, which guards it with a Mutex).
#[pymodule(gil_used = false)]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(test_function, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_size, m)?)?;
    m.add_function(wrap_pyfunction!(collect_tensors, m)?)?;
    m.add_function(wrap_pyfunction!(bytecode_printer::call_with_bytecode_printing, m)?)?;
    m.add_function(wrap_pyfunction!(cfg::python::bytecode_to_cfg, m)?)?;
    m.add_function(wrap_pyfunction!(exec::call_with_cfg_execution, m)?)?;
    m.add_function(wrap_pyfunction!(exec::_set_cfg_outline, m)?)?;
    m.add_function(wrap_pyfunction!(exec::_set_cfg_optimize, m)?)?;
    m.add_function(wrap_pyfunction!(exec::_set_cfg_record, m)?)?;
    m.add_function(wrap_pyfunction!(exec::cfg_observed, m)?)?;
    m.add_function(wrap_pyfunction!(exec::optimize_block, m)?)?;
    m.add_function(wrap_pyfunction!(exec::_specialize, m)?)?;
    m.add_function(wrap_pyfunction!(exec::_reoptimize, m)?)?;
    m.add_function(wrap_pyfunction!(exec::_cfg_clones, m)?)?;
    m.add_function(wrap_pyfunction!(exec::_set_cfg_elide, m)?)?;
    m.add_function(wrap_pyfunction!(exec::_add_block_checks, m)?)?;
    m.add_function(wrap_pyfunction!(exec::_cfg_misses, m)?)?;
    m.add_function(wrap_pyfunction!(exec::_cfg_region_stats, m)?)?;
    m.add_function(wrap_pyfunction!(exec::region::_set_cfg_regions, m)?)?;
    m.add_function(wrap_pyfunction!(exec::region::_cfg_region_passed_over, m)?)?;
    m.add_function(wrap_pyfunction!(exec::reads::_region_own, m)?)?;
    m.add_function(wrap_pyfunction!(exec::writes::_region_writes, m)?)?;
    m.add_function(wrap_pyfunction!(exec::writes::_set_cfg_effects, m)?)?;
    m.add_function(wrap_pyfunction!(exec::writes::_cfg_write_site, m)?)?;
    m.add_function(wrap_pyfunction!(exec::_cfg_plan_stats, m)?)?;
    m.add_function(wrap_pyfunction!(exec::region::_set_cfg_region_observe, m)?)?;
    m.add_function(wrap_pyfunction!(exec::region::_set_cfg_region_reads, m)?)?;
    m.add_function(wrap_pyfunction!(exec::_set_cfg_spans, m)?)?;
    m.add_function(wrap_pyfunction!(exec::_set_cfg_pure_calls, m)?)?;
    m.add_function(wrap_pyfunction!(exec::_set_cfg_pure_callables, m)?)?;
    m.add_function(wrap_pyfunction!(exec::_cfg_untraced, m)?)?;
    m.add_function(wrap_pyfunction!(exec::_set_cfg_reoptimize_after, m)?)?;
    m.add_function(wrap_pyfunction!(exec::_reoptimize_all, m)?)?;
    m.add_function(wrap_pyfunction!(exec::_cfg_bypass_cold, m)?)?;
    m.add_function(wrap_pyfunction!(exec::_cfg_skip, m)?)?;
    m.add_function(wrap_pyfunction!(exec::_cfg_impure, m)?)?;
    m.add_function(wrap_pyfunction!(exec::_set_cfg_pass_hook, m)?)?;
    m.add_function(wrap_pyfunction!(exec::_cfg_opaque, m)?)?;
    m.add_function(wrap_pyfunction!(exec::_cfg_opaque_site, m)?)?;
    m.add_function(wrap_pyfunction!(exec::_set_cfg_scalar_reads, m)?)?;
    m.add_class::<exec::TreeChecks>()?;
    m.add_function(wrap_pyfunction!(exec::region::_region_body, m)?)?;
    m.add_function(wrap_pyfunction!(exec::region::_region_refuse, m)?)?;
    m.add_function(wrap_pyfunction!(exec::reads::_region_reads, m)?)?;
    m.add_function(wrap_pyfunction!(exec::reads::_region_known, m)?)?;
    m.add_function(wrap_pyfunction!(exec::reads::_region_read_misses, m)?)?;
    m.add_class::<exec::reads::Reads>()?;
    m.add_class::<exec::overlap::Overlap>()?;
    m.add_function(wrap_pyfunction!(exec::graph::_run_aten_graph, m)?)?;
    m.add_class::<exec::graph::Ref>()?;
    m.add_function(wrap_pyfunction!(exec::joint::_trace_joint, m)?)?;
    m.add_function(wrap_pyfunction!(exec::autograd::_set_cfg_autograd, m)?)?;
    m.add_function(wrap_pyfunction!(exec::autograd::_cfg_last_joint, m)?)?;
    m.add_function(wrap_pyfunction!(exec::_cfg_autograd_blocks, m)?)?;
    m.add_function(wrap_pyfunction!(exec::torch_ops::set_print_torch_ops, m)?)?;
    m.add_function(wrap_pyfunction!(cfg::python::_cfg_opcodes, m)?)?;
    m.add_function(wrap_pyfunction!(cfg::python::_cfg_signature, m)?)?;
    Ok(())
}
