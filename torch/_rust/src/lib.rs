use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet, PyList, PySet, PyTuple};

mod bindings;

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
    if let Ok(t) = obj.downcast::<Tensor>() {
        out.push(t.clone());
        return Ok(());
    }
    if let Ok(d) = obj.downcast::<PyDict>() {
        for (_, v) in d.iter() {
            collect_into(&v, out)?;
        }
    } else if let Ok(seq) = obj.downcast::<PyList>() {
        for item in seq.iter() {
            collect_into(&item, out)?;
        }
    } else if let Ok(seq) = obj.downcast::<PyTuple>() {
        for item in seq.iter() {
            collect_into(&item, out)?;
        }
    } else if let Ok(s) = obj.downcast::<PySet>() {
        for item in s.iter() {
            collect_into(&item, out)?;
        }
    } else if let Ok(s) = obj.downcast::<PyFrozenSet>() {
        for item in s.iter() {
            collect_into(&item, out)?;
        }
    }
    Ok(())
}

// Declare the module safe to run without the GIL so the free-threaded CPython
// build (e.g. py3.14t) does not re-enable the GIL and emit a RuntimeWarning when
// torch._rust is imported. The functions here only do read-only access through
// pyo3's safe APIs and hold no shared mutable state; preserve that when adding
// new ones.
#[pymodule(gil_used = false)]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(test_function, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_size, m)?)?;
    m.add_function(wrap_pyfunction!(collect_tensors, m)?)?;
    Ok(())
}
