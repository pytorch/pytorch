//! The Python face of the crate: compiled functions, and the view of a region
//! handed to passes written in Python.

use crate::bytecode::{self, OpcodeTable};
use crate::capi::{self, TcfgApi, TCFG_API_VERSION};
use crate::frontend;
use crate::ir::*;
use crate::pyffi::PyObj;
use crate::spec::{self, CfgPass, PassCx, Pipeline, PyPass, Unit, ValueObs};
use crate::vm::{Compiled, Options};
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyCapsule, PyDict, PyList, PyTuple};
use std::cell::Cell;
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
pub fn builtin_passes() -> Vec<&'static str> {
    spec::BUILTIN_PASSES.to_vec()
}

#[pyfunction]
#[pyo3(signature = (func, passes, specialize=true, skip_autograd=true, unboxed=true, max_versions=4, max_respecializations=6))]
pub fn compile(
    py: Python<'_>,
    func: &Bound<'_, PyAny>,
    passes: Vec<Bound<'_, PyAny>>,
    specialize: bool,
    skip_autograd: bool,
    unboxed: bool,
    max_versions: usize,
    max_respecializations: u32,
) -> PyResult<CfgFunction> {
    let table = OpcodeTable::load(py)?;
    let function = frontend::build(py, &table, func)?;
    let mut pipeline: Vec<Box<dyn CfgPass>> = Vec::new();
    for pass in passes {
        if let Ok(name) = pass.extract::<String>() {
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
        } else if pass.is_callable() {
            let name = pass
                .getattr("__name__")
                .and_then(|n| n.extract::<String>())
                .unwrap_or_else(|_| "<python pass>".to_string());
            pipeline.push(Box::new(PyPass {
                callable: unsafe { PyObj::from_borrowed(pass.as_ptr()) },
                name,
            }));
        } else {
            return Err(PyValueError::new_err("a pass is a built-in pass name or a callable"));
        }
    }
    let options = Options {
        specialize,
        skip_autograd,
        unboxed,
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
        out.set_item("unboxed_results", cf.unboxed_results.load(Ordering::Relaxed))?;
        out.set_item("boxed_later", cf.boxed_later.load(Ordering::Relaxed))?;
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

/// The region a pass is transforming. Only valid while the pass runs.
#[pyclass(module = "torch._cfg_rs", name = "Region", unsendable)]
pub struct Region {
    unit: Cell<*mut Unit>,
    cx: Cell<*const PassCx<'static>>,
}

impl Region {
    fn parts(&self) -> PyResult<(&mut Unit, &PassCx<'static>)> {
        if self.unit.get().is_null() {
            return Err(PyRuntimeError::new_err("a Region is only valid while its pass runs"));
        }
        Ok(unsafe { (&mut *self.unit.get(), &*self.cx.get()) })
    }
}

#[pymethods]
impl Region {
    /// Index of the basic block this region covers.
    #[getter]
    fn block(&self) -> PyResult<usize> {
        Ok(self.parts()?.0.block)
    }

    /// Whether the pass may add guards based on what was observed.
    #[getter]
    fn speculate(&self) -> PyResult<bool> {
        Ok(self.parts()?.0.speculate)
    }

    fn __str__(&self) -> PyResult<String> {
        let (unit, cx) = self.parts()?;
        Ok(unit.dump(cx.func()))
    }

    /// The instructions as dicts with keys index, op, dst, uses and text.
    fn instructions(&self, py: Python<'_>) -> PyResult<PyObject> {
        let (unit, cx) = self.parts()?;
        let func = cx.func();
        let out = PyList::empty_bound(py);
        for (i, u) in unit.insts.iter().enumerate() {
            let d = PyDict::new_bound(py);
            let text = func.op_repr(&u.inst.op);
            d.set_item("index", i)?;
            d.set_item("op", text.split_whitespace().next().unwrap_or(""))?;
            d.set_item("dst", if u.inst.dst == NONE { None } else { Some(u.inst.dst) })?;
            let mut uses = Vec::new();
            u.inst.op.for_each_use(|v| uses.push(v));
            d.set_item("uses", uses)?;
            d.set_item("text", text)?;
            out.append(d)?;
        }
        Ok(out.into_any().unbind())
    }

    /// What the profiling tier saw the SSA value be, or None.
    fn observed(&self, py: Python<'_>, value: V) -> PyResult<PyObject> {
        let (_, cx) = self.parts()?;
        let Some(obs) = cx.obs.get(value as usize) else {
            return Err(PyIndexError::new_err("no such value"));
        };
        let d = PyDict::new_bound(py);
        match obs {
            ValueObs::Unseen => return Ok(py.None()),
            ValueObs::Tensor { min_rank, dims, last } => {
                d.set_item("kind", "tensor")?;
                d.set_item("rank", *min_rank)?;
                d.set_item("dims", dims.clone())?;
                d.set_item("last", last.clone())?;
            }
            ValueObs::Bool { seen_true, seen_false, last } => {
                d.set_item("kind", "bool")?;
                d.set_item("true", *seen_true)?;
                d.set_item("false", *seen_false)?;
                d.set_item("last", *last)?;
            }
            ValueObs::Int { lo, hi } => {
                d.set_item("kind", "int")?;
                d.set_item("lo", *lo)?;
                d.set_item("hi", *hi)?;
            }
            ValueObs::Other => d.set_item("kind", "other")?,
        }
        Ok(d.into_any().unbind())
    }

    /// Makes this version of the region depend on lo <= value.size(dim) <= hi
    /// (hi=None for no upper bound). The check goes to the region entry if
    /// the value exists there, else right after its definition. Returns True
    /// if the guard was added, False if the region already depended on it,
    /// and None if it cannot be placed.
    #[pyo3(signature = (value, dim, lo, hi=None))]
    fn guard_dim(&self, value: V, dim: i64, lo: i64, hi: Option<i64>) -> PyResult<Option<bool>> {
        let (unit, cx) = self.parts()?;
        if value as usize >= cx.obs.len() {
            return Err(PyIndexError::new_err("no such value"));
        }
        let cond = GuardCond::DimRange { v: value, dim, lo, hi: hi.unwrap_or(i64::MAX) };
        let before = unit.def_pos(value).map_or(0, |p| p + 1);
        Ok(unit.add_guard(cx, cond, before))
    }

    /// Replaces the instruction at `index` with the constant True, False or
    /// None.
    fn replace_with_const(&self, index: usize, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let (unit, cx) = self.parts()?;
        let func = cx.func();
        let c = if value.is_none() {
            func.const_none
        } else if let Ok(b) = value.downcast::<pyo3::types::PyBool>() {
            if b.is_true() {
                func.const_true
            } else {
                func.const_false
            }
        } else {
            return Err(PyValueError::new_err("only True, False and None can be folded to"));
        };
        let Some(u) = unit.insts.get_mut(index) else {
            return Err(PyIndexError::new_err("no such instruction"));
        };
        if u.inst.dst == NONE {
            return Err(PyValueError::new_err("the instruction does not produce a value"));
        }
        u.inst.op = Op::Const(c);
        Ok(())
    }
}

pub fn run_python_pass(callable: &PyObj, unit: &mut Unit, cx: &PassCx) -> PyResult<bool> {
    Python::with_gil(|py| {
        let region = Py::new(
            py,
            Region {
                unit: Cell::new(unit as *mut Unit),
                cx: Cell::new(cx as *const PassCx as *const PassCx<'static>),
            },
        )?;
        let callable = unsafe { Bound::from_borrowed_ptr(py, callable.as_ptr()) };
        let result = callable.call1((region.clone_ref(py),));
        let r = region.borrow(py);
        r.unit.set(null_mut());
        r.cx.set(std::ptr::null());
        result?.is_truthy()
    })
}
