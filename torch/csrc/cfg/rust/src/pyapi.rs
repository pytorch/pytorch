//! The Python face of the crate: compiled functions, and the view of a region
//! handed to passes written in Python.

use crate::bytecode::{self, OpcodeTable};
use crate::capi::{self, TcfgApi, TCFG_API_VERSION};
use crate::ir::*;
use crate::pyffi::PyObj;
use crate::spec::{self, CfgPass, PassCx, Pipeline, PyPass, Unit, ValueObs};
use crate::vm::{self, Callees, Compiled, Options};
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

fn interned(py: Python<'_>, name: &str) -> PyObj {
    unsafe { PyObj::from_borrowed(pyo3::types::PyString::intern_bound(py, name).as_ptr()) }
}

/// Registers that a call of an instance whose class has `function` as its
/// `__call__` is a call of the instance's method `target`, if every
/// condition holds. A condition is `(kind, a, b)`:
/// `("instance_falsy", name, None)`, `("attr_none", name, None)`,
/// `("global_falsy", dict, name)`, `("call_falsy", callable, None)`,
/// `("not_tracing", None, None)` or `("not_shadowed", name, None)`.
#[pyfunction]
pub fn register_call_shortcut(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    target: &str,
    conditions: Vec<(String, Bound<'_, PyAny>, Option<Bound<'_, PyAny>>)>,
) -> PyResult<()> {
    use crate::shortcuts::Condition;
    let object = |o: &Bound<'_, PyAny>| unsafe { PyObj::from_borrowed(o.as_ptr()) };
    let name = |o: &Bound<'_, PyAny>| -> PyResult<PyObj> { Ok(interned(py, &o.extract::<String>()?)) };
    let mut parsed = Vec::new();
    for (kind, a, b) in &conditions {
        parsed.push(match (kind.as_str(), b) {
            ("instance_falsy", None) => Condition::InstanceFalsy(name(a)?),
            ("attr_none", None) => Condition::AttrNone(name(a)?),
            ("global_falsy", Some(b)) => {
                Condition::GlobalFalsy(object(a), name(b)?, Cell::new((0, null_mut())))
            }
            ("call_falsy", None) => Condition::CallFalsy(object(a)),
            ("not_tracing", None) => Condition::NotTracing,
            ("not_shadowed", None) => Condition::NotShadowed(name(a)?),
            _ => return Err(PyValueError::new_err(format!("unknown shortcut condition {:?}", kind))),
        });
    }
    crate::shortcuts::register_call(crate::shortcuts::CallShortcut {
        function: object(function),
        target: interned(py, target),
        conditions: parsed,
    });
    Ok(())
}

/// Registers that, for an instance whose class has `function` as its
/// `__getattr__`, an attribute the normal lookup does not find is an entry
/// of one of the dicts that the instance's dict has under these names.
#[pyfunction]
pub fn register_getattr_shortcut(py: Python<'_>, function: &Bound<'_, PyAny>, dicts: Vec<String>) {
    crate::shortcuts::register_getattr(crate::shortcuts::GetattrShortcut {
        function: unsafe { PyObj::from_borrowed(function.as_ptr()) },
        dicts: dicts.iter().map(|name| interned(py, name)).collect(),
    });
}

#[pyfunction]
pub fn clear_shortcuts() {
    crate::shortcuts::clear();
}

#[pyfunction]
#[pyo3(signature = (func, passes, specialize=true, skip_autograd=true, unboxed=true, max_versions=4, max_respecializations=6, inline=true, unroll=true, callees=true, shortcuts=true, demote=true))]
pub fn compile(
    py: Python<'_>,
    func: &Bound<'_, PyAny>,
    passes: Vec<Bound<'_, PyAny>>,
    specialize: bool,
    skip_autograd: bool,
    unboxed: bool,
    max_versions: usize,
    max_respecializations: u32,
    inline: bool,
    unroll: bool,
    callees: bool,
    shortcuts: bool,
    demote: bool,
) -> PyResult<CfgFunction> {
    let table = OpcodeTable::load(py)?;
    let options = Options {
        specialize,
        skip_autograd,
        unboxed,
        max_versions: max_versions.max(1),
        max_respecializations,
        inline,
        unroll,
        callees,
        shortcuts,
        demote,
        ..Options::default()
    };
    let function = vm::lower(py, &table, func, &options)?;
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
    Ok(CfgFunction {
        inner: Arc::new(Compiled::new(
            function,
            options,
            Arc::new(Pipeline { passes: pipeline }),
            Arc::new(Callees::new(table)),
        )),
        wrapped: func.clone().unbind(),
    })
}

/// A function compiled to an SSA control-flow graph. Calling it runs the
/// graph in the VM.
#[pyclass(module = "torch._cfg_rs", name = "CfgFunction", frozen, weakref)]
pub struct CfgFunction {
    inner: Arc<Compiled>,
    wrapped: PyObject,
}

impl Drop for CfgFunction {
    fn drop(&mut self) {
        // What was compiled under this function refers back to the table.
        self.inner.callees.clear();
    }
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
            unsafe { self.inner.call_measured(self.wrapped.as_ptr(), args.as_ptr(), kwargs.map_or(null_mut(), |k| k.as_ptr())) };
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

    /// Everything this object does not define itself is read from the
    /// function it was compiled from: `__code__`, `__defaults__`, `__dict__`
    /// and what decorators stored in it.
    fn __getattr__(&self, py: Python<'_>, name: &Bound<'_, pyo3::types::PyString>) -> PyResult<PyObject> {
        self.wrapped.bind(py).getattr(name).map(|v| v.unbind())
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
        out.set_item("calls", cf.calls())?;
        let blocks = PyList::empty_bound(py);
        let unlowered = PyList::empty_bound(py);
        let (mut sites, mut hits, mut misses) = (0u64, 0u64, 0u64);
        for (b, bs) in state.blocks.iter().enumerate() {
            let d = PyDict::new_bound(py);
            d.set_item("block", b)?;
            let (entries, specialized) = cf.block_entries(b, bs);
            d.set_item("executions", entries)?;
            d.set_item("specialized_executions", specialized)?;
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
        let inlined: Vec<String> = cf.func.origins[1..].iter().map(|o| o.name.clone()).collect();
        out.set_item("inlined", inlined)?;
        out.set_item("unrolled_loops", cf.func.unrolled)?;
        let (compiled, interpreted) = cf.callees.summary();
        out.set_item("compiled_callees", compiled)?;
        out.set_item("interpreted_callees", interpreted)?;
        out.set_item("demoted_callees", cf.callees.demoted())?;
        out.set_item("demoted", cf.is_demoted())?;
        out.set_item("measuring", cf.is_measuring())?;
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

    /// The file the function of the block is defined in.
    #[getter]
    fn filename(&self) -> PyResult<String> {
        Ok(self.parts()?.1.func().filename.clone())
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
            d.set_item("replayable", u.origin as usize == i && u.inst.op.is_replayable())?;
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

    /// What a pass left on this block with `Replay.settle`, or None.
    #[getter]
    fn note(&self, py: Python<'_>) -> PyResult<PyObject> {
        let (unit, cx) = self.parts()?;
        let state = cx.cf.state.borrow();
        Ok(match &state.blocks[unit.block].note {
            Some(note) => unsafe { PyObject::from_borrowed_ptr(py, note.as_ptr()) },
            None => py.None(),
        })
    }

    /// Leaves an object on the block for the next time a pass sees it.
    #[setter]
    fn set_note(&self, note: &Bound<'_, PyAny>) -> PyResult<()> {
        let (unit, cx) = self.parts()?;
        cx.cf.state.borrow_mut().blocks[unit.block].note =
            if note.is_none() { None } else { Some(unsafe { PyObj::from_borrowed(note.as_ptr()) }) };
        Ok(())
    }

    /// A handle on the generic instructions start..end of the block, which
    /// outlives the pass: it can run them again on other values, and says
    /// which values they take and give.
    fn replayer(&self, start: usize, end: usize) -> PyResult<Replay> {
        let (unit, cx) = self.parts()?;
        let count = cx.func().blocks[unit.block].insts.len();
        if start >= end || end > count {
            return Err(PyIndexError::new_err("no such range of instructions"));
        }
        // Every Compiled lives in an Arc.
        let compiled = unsafe {
            let pointer = cx.cf as *const Compiled;
            std::sync::Arc::increment_strong_count(pointer);
            let owned = std::sync::Arc::from_raw(pointer);
            std::sync::Arc::downgrade(&owned)
        };
        Ok(Replay { compiled, block: unit.block, start, end })
    }

    /// Replaces the instructions start..end, which must still be the generic
    /// ones, by a call of `callable` with the values the range takes from
    /// before it (see `Replay.io`), which returns a tuple of the values it
    /// gives. Returns False if that cannot be done.
    fn fuse(&self, start: usize, end: usize, callable: &Bound<'_, PyAny>) -> PyResult<bool> {
        let (unit, cx) = self.parts()?;
        if start >= end || end > unit.insts.len() {
            return Err(PyIndexError::new_err("no such range of instructions"));
        }
        if unit.insts[start..end].iter().enumerate().any(|(i, u)| u.origin as usize != start + i) {
            return Ok(false);
        }
        let (inputs, outputs) = cx.cf.range_io(unit.block, start, end);
        let tuple = cx.cf.scratch();
        let bc = unit.insts[start].inst.bc;
        // What the range held alive up to some point, it now holds to its
        // end: the values from before the range, and the ones it gives.
        let mut defined: Vec<V> = Vec::new();
        let mut held: Vec<V> = Vec::new();
        for u in &unit.insts[start..end] {
            if let Op::Hold(values) = &u.inst.op {
                for v in values {
                    if (!defined.contains(v) || outputs.contains(v)) && !held.contains(v) {
                        held.push(*v);
                    }
                }
            }
            if u.inst.dst != NONE {
                defined.push(u.inst.dst);
            }
        }
        let callee = unsafe { PyObj::from_borrowed(callable.as_ptr()) };
        let mut fused = vec![crate::spec::UInst {
            inst: Inst { dst: tuple, op: Op::CallObject(callee, inputs), bc, kills: Vec::new() },
            origin: start as u32,
            removable: false,
        }];
        for (i, output) in outputs.iter().enumerate() {
            fused.push(crate::spec::UInst {
                inst: Inst { dst: *output, op: Op::TupleGet(tuple, i as u32), bc, kills: Vec::new() },
                origin: NONE,
                removable: false,
            });
        }
        if !held.is_empty() {
            fused.push(crate::spec::UInst {
                inst: Inst { dst: NONE, op: Op::Hold(held), bc, kills: Vec::new() },
                origin: NONE,
                removable: false,
            });
        }
        unit.insts.splice(start..end, fused);
        Ok(true)
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

/// A range of the generic instructions of a block, to be run again on other
/// values. Made by `Region.replayer`.
#[pyclass(module = "torch._cfg_rs", name = "Replay", frozen)]
pub struct Replay {
    compiled: std::sync::Weak<Compiled>,
    block: usize,
    start: usize,
    end: usize,
}

impl Replay {
    fn compiled(&self) -> PyResult<std::sync::Arc<Compiled>> {
        self.compiled
            .upgrade()
            .ok_or_else(|| PyRuntimeError::new_err("the compiled function no longer exists"))
    }

    fn check(&self, lo: usize, hi: usize) -> PyResult<()> {
        if lo < self.start || hi > self.end || lo > hi {
            return Err(PyIndexError::new_err("outside the range of the replay"));
        }
        Ok(())
    }
}

#[pymethods]
impl Replay {
    #[getter]
    fn block(&self) -> usize {
        self.block
    }

    #[getter]
    fn start(&self) -> usize {
        self.start
    }

    #[getter]
    fn end(&self) -> usize {
        self.end
    }

    /// Name and line of the function, for messages.
    #[getter]
    fn name(&self) -> PyResult<String> {
        let compiled = self.compiled()?;
        Ok(format!("{} (block {})", compiled.func.qualname, self.block))
    }

    /// The values the instructions lo..hi take from before them, and the
    /// ones they define that are needed afterwards, as two lists.
    fn io(&self, lo: usize, hi: usize) -> PyResult<(Vec<V>, Vec<V>)> {
        self.check(lo, hi)?;
        Ok(self.compiled()?.range_io(self.block, lo, hi))
    }

    /// The instructions lo..hi as text.
    fn text(&self, lo: usize, hi: usize) -> PyResult<Vec<String>> {
        self.check(lo, hi)?;
        let compiled = self.compiled()?;
        let func = &compiled.func;
        Ok(func.blocks[self.block].insts[lo..hi].iter().map(|inst| func.op_repr(&inst.op)).collect())
    }

    /// Runs the instructions lo..hi on the values in `env`, a dict from value
    /// numbers to objects; afterwards `env` holds the values still needed.
    /// With `traced`, what the instructions call runs in the interpreter
    /// instead of compiled. An exception gets the index of the instruction
    /// that raised it as its attribute `cfg_index`.
    #[pyo3(signature = (lo, hi, env, traced=false))]
    fn run(&self, py: Python<'_>, lo: usize, hi: usize, env: &Bound<'_, PyDict>, traced: bool) -> PyResult<()> {
        self.check(lo, hi)?;
        let compiled = self.compiled()?;
        match unsafe { compiled.replay(self.block, lo, hi, env.as_ptr(), traced) } {
            Ok(()) => Ok(()),
            Err(index) => {
                let err = PyErr::fetch(py);
                let _ = err.value_bound(py).setattr("cfg_index", index);
                Err(err)
            }
        }
    }

    /// Leaves `note` on the block for the pass and has the block specialized
    /// again, with that note, before this returns.
    fn settle(&self, note: &Bound<'_, PyAny>) -> PyResult<()> {
        let compiled = self.compiled()?;
        compiled.state.borrow_mut().blocks[self.block].note =
            Some(unsafe { PyObj::from_borrowed(note.as_ptr()) });
        compiled.respecialize(self.block);
        Ok(())
    }
}

/// Tells the VM that compiled code changed behind its back: measurements of
/// compiled code against the interpreter that are under way start over.
#[pyfunction]
pub fn plan_changed() {
    vm::plan_changed();
}

/// How many blocks on the calling thread are in their profiling run.
#[pyfunction]
pub fn profiling_depth() -> usize {
    vm::profiling_depth()
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
