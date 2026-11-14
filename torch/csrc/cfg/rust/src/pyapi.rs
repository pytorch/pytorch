//! The Python face of the crate: compiled functions.

use crate::bytecode::{self, OpcodeTable};
use crate::frontend;
use crate::ir::Function;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple};

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

#[pyfunction]
pub fn compile(py: Python<'_>, func: &Bound<'_, PyAny>) -> PyResult<CfgFunction> {
    let table = OpcodeTable::load(py)?;
    Ok(CfgFunction { func: frontend::build(py, &table, func)?, wrapped: func.clone().unbind() })
}

/// A function compiled to an SSA control-flow graph.
#[pyclass(module = "torch._cfg_rs", name = "CfgFunction", frozen)]
pub struct CfgFunction {
    func: Function,
    wrapped: PyObject,
}

#[pymethods]
impl CfgFunction {
    /// Nothing executes the graph yet: a call runs the original function.
    #[pyo3(signature = (*args, **kwargs))]
    fn __call__(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyObject> {
        self.wrapped.call_bound(py, args.clone(), kwargs)
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
        self.func.name.clone()
    }

    #[getter]
    fn __qualname__(&self) -> String {
        self.func.qualname.clone()
    }

    fn __repr__(&self) -> String {
        format!("<cfg_compile function {}>", self.func.qualname)
    }

    /// The SSA control-flow graph as text.
    fn dump(&self) -> String {
        self.func.dump()
    }
}
