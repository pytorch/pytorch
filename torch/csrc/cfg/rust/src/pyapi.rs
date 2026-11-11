//! The Python face of the crate: compiled functions.

use crate::bytecode::OpcodeTable;
use crate::capi::{self, TcfgApi, TCFG_API_VERSION};
use crate::frontend;
use crate::spec::{self, CfgPass, Pipeline};
use crate::vm::{Compiled, Options};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict, PyList, PyTuple};
use std::ptr::null_mut;
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// Hands the C function table of torch._C to the VM.
#[pyfunction]
pub fn init(capsule: &Bound<'_, PyCapsule>) -> PyResult<()> {
    let api = capsule.pointer() as *mut TcfgApi;
    if api.is_null() || unsafe { (*api).version } != TCFG_API_VERSION {
        return Err(PyRuntimeError::new_err(
            "torch._C and the cfg_compile runtime were built from different sources",
        ));
    }
    capi::set_api(api);
    Ok(())
}

#[pyfunction]
pub fn builtin_passes() -> Vec<&'static str> {
    spec::BUILTIN_PASSES.to_vec()
}

#[pyfunction]
#[pyo3(signature = (func, passes, specialize=true, skip_autograd=true, max_versions=4, max_respecializations=6))]
pub fn compile(
    py: Python<'_>,
    func: &Bound<'_, PyAny>,
    passes: Vec<String>,
    specialize: bool,
    skip_autograd: bool,
    max_versions: usize,
    max_respecializations: u32,
) -> PyResult<CfgFunction> {
    let table = OpcodeTable::load(py)?;
    let function = frontend::build(py, &table, func)?;
    let mut pipeline: Vec<Box<dyn CfgPass>> = Vec::new();
    for name in passes {
        match spec::builtin_pass(&name) {
            Some(p) => pipeline.push(p),
            None => {
                return Err(PyValueError::new_err(format!(
                    "unknown pass '{}'; built-in passes are {:?}",
                    name,
                    spec::BUILTIN_PASSES
                )))
            }
        }
    }
    let options = Options {
        specialize,
        skip_autograd,
        max_versions: max_versions.max(1),
        max_respecializations,
        ..Options::default()
    };
    Ok(CfgFunction {
        inner: Arc::new(Compiled::new(function, options, Pipeline { passes: pipeline })),
        wrapped: func.clone().unbind(),
    })
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

    /// The names of the passes this function is specialized with, in order.
    #[getter]
    fn passes(&self) -> Vec<String> {
        self.inner.pipeline.names()
    }

    /// The SSA control-flow graph as text: the generic code of every block
    /// followed by its specialized versions.
    #[pyo3(signature = (versions=true))]
    fn dump(&self, versions: bool) -> String {
        let cf = &self.inner;
        let func = &cf.func;
        if !versions {
            return func.dump();
        }
        let mut out = String::new();
        let params: Vec<String> = func.params.iter().map(|v| func.value_name(*v)).collect();
        out.push_str(&format!(
            "function {}({})    ; {}:{}\n",
            func.qualname,
            params.join(", "),
            func.filename,
            func.firstlineno
        ));
        let state = cf.state.borrow();
        for b in 0..func.blocks.len() {
            func.dump_block_header(&mut out, b);
            func.dump_code(&mut out, &func.blocks[b].insts, &func.blocks[b].term);
            for version in &state.blocks[b].versions {
                let guards: Vec<String> =
                    version.entry_guards.iter().map(|g| func.guard_repr(g)).collect();
                out.push_str(&format!(
                    "  b{} specialized, version {}{}\n",
                    b,
                    version.id,
                    if guards.is_empty() {
                        String::new()
                    } else {
                        format!("    ; if {}", guards.join(" and "))
                    }
                ));
                func.dump_code(&mut out, &version.insts, &version.term);
            }
        }
        out
    }

    /// Counters of the tiers, per block, and the instructions that made
    /// dispatcher calls but could not be turned into direct kernel calls.
    fn stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let cf = &self.inner;
        let state = cf.state.borrow();
        let out = PyDict::new_bound(py);
        out.set_item("calls", state.calls)?;
        let blocks = PyList::empty_bound(py);
        let unlowered = PyList::empty_bound(py);
        let (mut sites, mut hits, mut misses) = (0u64, 0u64, 0u64);
        for (b, bs) in state.blocks.iter().enumerate() {
            let d = PyDict::new_bound(py);
            d.set_item("block", b)?;
            d.set_item("executions", bs.execs)?;
            d.set_item("specialized_executions", bs.spec_execs)?;
            d.set_item("profile_runs", bs.profile_runs)?;
            d.set_item("versions", bs.versions.len())?;
            d.set_item("entry_guard_misses", bs.entry_misses)?;
            d.set_item("guard_failures", bs.deopts)?;
            d.set_item("rejected", bs.rejected.clone())?;
            blocks.append(d)?;
            for (i, slot) in bs.slots.iter().enumerate() {
                if let Some(slot) = slot {
                    sites += 1;
                    hits += slot.hits.load(Ordering::Relaxed);
                    misses += slot.misses.load(Ordering::Relaxed);
                } else if let Some(reason) = &bs.reasons[i] {
                    let text = cf.func.op_repr(&cf.func.blocks[b].insts[i].op);
                    unlowered.append((b, i, text, reason.clone()))?;
                }
            }
        }
        out.set_item("blocks", blocks)?;
        out.set_item("sites", sites)?;
        out.set_item("site_hits", hits)?;
        out.set_item("site_misses", misses)?;
        out.set_item("unlowered", unlowered)?;
        out.set_item("log", state.log.clone())?;
        Ok(out.into_any().unbind())
    }

    /// The dispatch sites: for every instruction that calls kernels directly,
    /// its block, index, text, and one description per captured variant.
    fn sites(&self, py: Python<'_>) -> PyResult<PyObject> {
        let cf = &self.inner;
        let slots: Vec<(usize, usize, Arc<capi::SiteSlot>)> = {
            let state = cf.state.borrow();
            let mut v = Vec::new();
            for (b, bs) in state.blocks.iter().enumerate() {
                for (i, slot) in bs.slots.iter().enumerate() {
                    if let Some(slot) = slot {
                        v.push((b, i, slot.clone()));
                    }
                }
            }
            v
        };
        let out = PyList::empty_bound(py);
        for (b, i, slot) in slots {
            let d = PyDict::new_bound(py);
            d.set_item("block", b)?;
            d.set_item("index", i)?;
            d.set_item("instruction", cf.func.op_repr(&cf.func.blocks[b].insts[i].op))?;
            d.set_item("hits", slot.hits.load(Ordering::Relaxed))?;
            d.set_item("misses", slot.misses.load(Ordering::Relaxed))?;
            let variants = unsafe { PyObject::from_owned_ptr(py, slot.info()) };
            d.set_item("variants", variants)?;
            out.append(d)?;
        }
        Ok(out.into_any().unbind())
    }
}
