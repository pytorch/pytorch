//! The Python face of the crate: compiled functions.

use crate::bytecode::OpcodeTable;
use crate::frontend;
use crate::vm::Compiled;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::ptr::null_mut;
use std::sync::Arc;

#[pyfunction]
pub fn compile(py: Python<'_>, func: &Bound<'_, PyAny>) -> PyResult<CfgFunction> {
    let table = OpcodeTable::load(py)?;
    let function = frontend::build(py, &table, func)?;
    Ok(CfgFunction { inner: Arc::new(Compiled::new(function)), wrapped: func.clone().unbind() })
}

/// A function compiled to an SSA control-flow graph. Calling it runs the
/// graph in the VM.
#[pyclass(module = "torch._cfg_rs", name = "CfgFunction", frozen)]
pub struct CfgFunction {
    inner: Arc<Compiled>,
    wrapped: PyObject,
}

#[pymethods]
impl CfgFunction {
    #[pyo3(signature = (*args, **kwargs))]
    fn __call__(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyObject> {
        let result =
            unsafe { self.inner.call(args.as_ptr(), kwargs.map_or(null_mut(), |k| k.as_ptr())) };
        if result.is_null() {
            Err(PyErr::fetch(py))
        } else {
            Ok(unsafe { PyObject::from_owned_ptr(py, result) })
        }
    }

    fn __get__(
        slf: &Bound<'_, Self>,
        obj: Option<&Bound<'_, PyAny>>,
        _owner: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        let py = slf.py();
        match obj {
            Some(obj) if !obj.is_none() => unsafe {
                let method = crate::pyffi::PyMethod_New(slf.as_ptr(), obj.as_ptr());
                if method.is_null() {
                    Err(PyErr::fetch(py))
                } else {
                    Ok(PyObject::from_owned_ptr(py, method))
                }
            },
            _ => Ok(slf.clone().into_any().unbind()),
        }
    }

    #[getter]
    fn __wrapped__(&self, py: Python<'_>) -> PyObject {
        self.wrapped.clone_ref(py)
    }

    #[getter]
    fn __name__(&self) -> String {
        self.inner.func.name.clone()
    }

    #[getter]
    fn __qualname__(&self) -> String {
        self.inner.func.qualname.clone()
    }

    fn __repr__(&self) -> String {
        format!("<cfg_compile function {}>", self.inner.func.qualname)
    }

    /// The SSA control-flow graph as text.
    fn dump(&self) -> String {
        self.inner.func.dump()
    }
}
