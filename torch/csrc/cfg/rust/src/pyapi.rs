//! The Python face of the crate: compiled functions, and the view of a region
//! handed to passes written in Python.

use crate::bytecode::{self, OpcodeTable};
use crate::capi::{self, TcfgApi, TCFG_API_VERSION};
use crate::graph::{GraphSite, GraphState, Graphs};
use crate::ir::*;
use crate::pyffi::{Obj, PyObj};
use crate::spec::{self, CalleeFact, CalleeKind, CfgPass, InstFacts, Kernels, PassCx, Pipeline, PyPass, Unit, ValueObs};
use crate::vm::{self, Callees, Compiled, Options};
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyCapsule, PyDict, PyList, PyTuple};
use std::cell::Cell;
use std::ffi::c_void;
use std::os::raw::c_char;
use std::ptr::null_mut;
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// Hands the C function table of torch._C to the VM.
#[pyfunction]
pub fn init(capsule: &Bound<'_, PyCapsule>) -> PyResult<()> {
    let api = capsule.pointer_checked(Some(c"torch._C._cfg_c_api"))?.as_ptr() as *mut TcfgApi;
    if unsafe { (*api).version } != TCFG_API_VERSION {
        return Err(PyRuntimeError::new_err(
            "torch._C and the torch.vm runtime were built from different sources",
        ));
    }
    capi::set_api(api);
    if let Some(hook) = unsafe { (*api).frame_hook } {
        unsafe { hook(vm::take_frame) };
    }
    Ok(())
}

/// The instructions of a code object, as (offset, opcode name, argument, jump
/// target) tuples. The frontend sees them after `bytecode::simplify`.
#[pyfunction]
pub fn decode(
    py: Python<'_>,
    code: &Bound<'_, PyAny>,
) -> PyResult<Vec<(u32, String, u32, Option<u32>)>> {
    let table = OpcodeTable::load(py)?;
    let co_code = code.getattr("co_code")?;
    let co_code = co_code.cast::<PyBytes>()?;
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
    unsafe { PyObj::from_borrowed(pyo3::types::PyString::intern(py, name).as_ptr()) }
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

/// Whether a Python function that C code called now, on this thread, would
/// be run in its compiled form: a compiled function is running, and what
/// runs is its own code, not the runtime's.
#[pyfunction]
pub fn frames_offered() -> bool {
    vm::frames_offered()
}

/// Leaves the functions whose code is in one of these files, or under one of
/// these directories (with a trailing separator), to the interpreter, with
/// all they call; `file::qualname` names one function of a file.
#[pyfunction]
pub fn leave_to_interpreter(places: Vec<String>) {
    vm::leave_to_interpreter(places);
}

#[pyfunction]
pub fn clear_shortcuts() {
    crate::shortcuts::clear();
}

#[pyfunction]
#[pyo3(signature = (func, passes, specialize=true, skip_autograd=true, unboxed=true, max_versions=4, max_respecializations=6, inline=true, unroll=true, max_trip_count=8, callees=true, shortcuts=true, demote=true, workspace=None, hook=true, graphs=None, export=None))]
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
    max_trip_count: i64,
    callees: bool,
    shortcuts: bool,
    demote: bool,
    workspace: Option<Bound<'_, PyAny>>,
    hook: bool,
    graphs: Option<(usize, usize)>,
    export: Option<Bound<'_, PyAny>>,
) -> PyResult<CfgFunction> {
    // What the pass "graphs" is made with: the least number of operations
    // of a graph and the most exported forms of one; and what makes an
    // exported form of the description of a graph (it returns the callable
    // and the guards of the passes, or the reason there is none), if the
    // graphs are to be exported at all.
    // What `torch._C._cfg_workspace` made: the memory the function keeps
    // between its calls.
    let mut memory = 0usize;
    if let Some(capsule) = &workspace {
        let pointer =
            unsafe { ffi::PyCapsule_GetPointer(capsule.as_ptr(), c"torch._C._cfg_workspace".as_ptr()) };
        if pointer.is_null() {
            return Err(PyErr::fetch(py));
        }
        memory = pointer as usize;
    }
    // From the first function on: see `chunks`.
    crate::chunks::keep();
    let table = OpcodeTable::load(py)?;
    let options = Options {
        specialize,
        skip_autograd,
        unboxed,
        max_versions: max_versions.max(1),
        max_respecializations,
        inline,
        unroll,
        max_trip_count,
        callees,
        shortcuts,
        demote,
        hook: hook && callees && capi::api().is_some_and(|api| api.frame_want.is_some()),
        ..Options::default()
    };
    let function = vm::lower(py, &table, func, &options)?;
    let mut pipeline: Vec<Box<dyn CfgPass>> = Vec::new();
    for pass in passes {
        if let Ok(name) = pass.extract::<String>() {
            if name == "graphs" {
                let made = Graphs::default();
                let (min_ops, max_variants) = graphs.unwrap_or((made.min_ops, made.max_variants));
                let export = export.as_ref().map(|e| unsafe { PyObj::from_borrowed(e.as_ptr()) });
                pipeline.push(Box::new(Graphs { min_ops, max_variants, export }));
                continue;
            }
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
        workspace: workspace.map(|w| w.unbind()),
        memory,
    })
}

/// A function compiled to an SSA control-flow graph. Calling it runs the
/// graph in the VM.
#[pyclass(module = "torch._cfg_rs", name = "CfgFunction", frozen, weakref)]
pub struct CfgFunction {
    inner: Arc<Compiled>,
    wrapped: Py<PyAny>,
    /// The capsule of the function's workspace and the address in it, or
    /// none and 0.
    workspace: Option<Py<PyAny>>,
    memory: usize,
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
    ) -> PyResult<Py<PyAny>> {
        // For the time of the call, tensors get their memory from what the
        // function kept of its earlier calls.
        let memory = self.memory as *mut std::ffi::c_void;
        let api = if memory.is_null() { None } else { capi::api() };
        let previous = api.map(|api| unsafe { (api.workspace_enter)(memory) });
        let result =
            unsafe { self.inner.call_measured(self.wrapped.as_ptr(), args.as_ptr(), kwargs.map_or(null_mut(), |k| k.as_ptr())) };
        if let (Some(api), Some(previous)) = (api, previous) {
            unsafe { (api.workspace_leave)(memory, previous) };
        }
        if result.is_null() {
            Err(PyErr::fetch(py))
        } else {
            Ok(unsafe { Bound::from_owned_ptr(py, result) }.unbind())
        }
    }

    fn __get__(
        slf: &Bound<'_, Self>,
        obj: Option<&Bound<'_, PyAny>>,
        _owner: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        match obj {
            Some(obj) if !obj.is_none() => unsafe {
                let method = crate::pyffi::PyMethod_New(slf.as_ptr(), obj.as_ptr());
                if method.is_null() {
                    Err(PyErr::fetch(py))
                } else {
                    Ok(Bound::from_owned_ptr(py, method).unbind())
                }
            },
            _ => Ok(slf.clone().into_any().unbind()),
        }
    }

    #[getter]
    fn __wrapped__(&self, py: Python<'_>) -> Py<PyAny> {
        self.wrapped.clone_ref(py)
    }

    /// The workspace the function was compiled with, if any.
    #[getter]
    fn workspace(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.workspace.as_ref().map(|w| w.clone_ref(py))
    }

    /// Everything this object does not define itself is read from the
    /// function it was compiled from: `__code__`, `__defaults__`, `__dict__`
    /// and what decorators stored in it.
    fn __getattr__(&self, py: Python<'_>, name: &Bound<'_, pyo3::types::PyString>) -> PyResult<Py<PyAny>> {
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
        format!("<torch.vm function {}>", self.inner.func.qualname)
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
                // Where graphs are exported, a block is shown as it runs
                // while they are there: with what enters them.
                let entered = |site: &Arc<GraphSite>| site.exporter.is_some();
                let entering = version.entering.as_ref().filter(|copy| copy.graphs.iter().any(entered));
                let insts = entering.map_or(&version.insts, |copy| &copy.insts);
                func.dump_code(&mut out, insts, &version.term);
            }
        }
        out
    }

    /// Counters of the tiers, per block, and the instructions that made
    /// dispatcher calls but could not be turned into direct kernel calls.
    fn stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let cf = &self.inner;
        let state = cf.state.borrow();
        let out = PyDict::new(py);
        out.set_item("calls", cf.calls())?;
        let blocks = PyList::empty(py);
        let unlowered = PyList::empty(py);
        let (mut sites, mut hits, mut misses) = (0u64, 0u64, 0u64);
        for (b, bs) in state.blocks.iter().enumerate() {
            let d = PyDict::new(py);
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
        drop(state);
        // The graphs of the function and of what it called: how many there
        // are, how many times one was handed to the passes over graphs, how
        // many calls ran what those made (`hits`) and how many came to a
        // graph and ran its instructions (`misses`), and how many graphs
        // are no longer exported.
        let (mut built, mut exported, mut hits, mut misses, mut given_up) = (0usize, 0u64, 0u64, 0u64, 0usize);
        let held = self.held();
        for (cf, _, _, site) in self.graph_sites() {
            // One whose operations a graph of a caller holds is no graph
            // of its own.
            if site.outer().is_some() || site.short().is_some() || Self::holder(&held, &cf, &site).is_some() {
                continue;
            }
            built += 1;
            exported += site.exports.load(Ordering::Relaxed);
            given_up += matches!(*site.state.borrow(), GraphState::GivenUp(_)) as usize;
            hits += site.hits.load(Ordering::Relaxed);
            misses += site.misses.load(Ordering::Relaxed);
        }
        let graphs = PyDict::new(py);
        graphs.set_item("built", built)?;
        graphs.set_item("exported", exported)?;
        graphs.set_item("hits", hits)?;
        graphs.set_item("misses", misses)?;
        graphs.set_item("given_up", given_up)?;
        out.set_item("graphs", graphs)?;
        Ok(out.into_any().unbind())
    }

    /// The dispatch sites: for every instruction that calls kernels directly,
    /// its block, index, text, and one description per captured variant.
    fn sites(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
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
        let out = PyList::empty(py);
        for (b, i, slot) in slots {
            let d = PyDict::new(py);
            d.set_item("block", b)?;
            d.set_item("index", i)?;
            d.set_item("instruction", cf.func.op_repr(&cf.func.blocks[b].insts[i].op))?;
            d.set_item("hits", slot.hits.load(Ordering::Relaxed))?;
            d.set_item("misses", slot.misses.load(Ordering::Relaxed))?;
            let variants = unsafe { Bound::from_owned_ptr(py, slot.info()) };
            d.set_item("variants", variants)?;
            out.append(d)?;
        }
        Ok(out.into_any().unbind())
    }

    /// What the profiling tier saw the instructions of the function do: for
    /// every instruction that ran there, its block, index and text, and
    /// `runs`, `kernels` ("none", "one" or "many" dispatcher calls; None
    /// where they were not counted), `frames` (a Python frame was entered),
    /// `raised`, `callee` (the Python function a call entered: its `name`,
    /// `code` and `kind`, "function", "method" or "instance"; "many" for
    /// more than one; None for none) and `wrote` (it wrote to a tensor it
    /// was given, in place).
    fn facts(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let cf = &self.inner;
        let found: Vec<(usize, usize, InstFacts)> = {
            let state = cf.state.borrow();
            let blocks = state.blocks.iter().enumerate();
            blocks
                .flat_map(|(b, bs)| bs.facts.iter().enumerate().map(move |(i, f)| (b, i, f.clone())))
                .filter(|(_, _, f)| f.runs > 0)
                .collect()
        };
        let out = PyList::empty(py);
        for (b, i, facts) in found {
            let d = facts_dict(py, &facts)?;
            d.set_item("block", b)?;
            d.set_item("index", i)?;
            d.set_item("instruction", cf.func.op_repr(&cf.func.blocks[b].insts[i].op))?;
            out.append(d)?;
        }
        Ok(out.into_any().unbind())
    }

    /// The graphs of the function and of the functions it called, and what
    /// was left out of them. One dict per graph, with `where` (function,
    /// block, and the generic instructions of the block from its first
    /// operation to the last of its run there), `state` ("built": the VM
    /// runs it; "waiting": it is exported at its next call; "compiled":
    /// what the passes over it made runs in its place, with a `reason` if
    /// it is not exported for further kinds of calls; "vm" with a `reason`:
    /// it is not exported, or not for the calls it has seen), `ops` and
    /// `operators` of the path that ran last, `inputs` and `outputs` (how
    /// many values that takes and gives), `branches` (how many branches and
    /// loop tests it goes through) and `calls` (how many calls of Python
    /// functions it goes into), `within` (for a graph of a function whose
    /// operations a path of a caller holds: the caller's graph, which is
    /// the one that runs them), `paths` (how many paths were taken
    /// down), `ends` (per path, the block and instruction the function goes
    /// on at behind it) and `why` (the last one ends there), `variants` (how
    /// many exported forms it has), `hits` and `misses` (how many calls ran
    /// an exported form, and how many came to the graph and ran its
    /// instructions), `exports` (how many times it was handed to the passes
    /// over graphs), `version` (of the block), and `forms`: per kind of call
    /// it was recorded for, the one that ran last first, `checks` (what a
    /// call has to be like, as text), `hits`, and the `reason` if calls of
    /// the kind are not exported. And one per instruction or run of
    /// instructions with tensor work that is in no graph, with `state` "vm",
    /// the `reason`, and for one instruction its `text`. A graph that
    /// starts where the path of another went through is part of that one,
    /// and not listed.
    fn graphs(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let out = PyList::empty(py);
        let mut known: Vec<(String, String)> = Vec::new();
        let mut shown: Vec<*const GraphSite> = Vec::new();
        let mut left = |py: Python<'_>, place: String, reason: String, text: Option<String>| -> PyResult<()> {
            let entry = (place, reason);
            if known.contains(&entry) {
                return Ok(());
            }
            let d = PyDict::new(py);
            d.set_item("where", entry.0.clone())?;
            d.set_item("state", "vm")?;
            d.set_item("reason", entry.1.clone())?;
            if let Some(text) = text {
                d.set_item("text", text)?;
            }
            known.push(entry);
            out.append(d)
        };
        let mut functions = vec![self.inner.clone()];
        functions.extend(self.inner.callees.compiled());
        let held = self.held();
        for cf in &functions {
            let func = &cf.func;
            let holds = |b: usize, lo: usize, hi: usize| {
                let insts = &func.blocks[b].insts[lo..hi.min(func.blocks[b].insts.len())];
                insts.iter().any(|inst| held.iter().any(|(owner, bc, ..)| *owner == Arc::as_ptr(cf) && *bc == inst.bc))
            };
            let versions: Vec<(usize, Arc<BlockCode>)> = {
                let state = cf.state.borrow();
                let blocks = state.blocks.iter().enumerate();
                blocks.flat_map(|(b, bs)| bs.versions.iter().map(move |version| (b, version.clone()))).collect()
            };
            let place = |b: usize, lo: usize, hi: usize| {
                if hi == lo + 1 {
                    (format!("{} (block {})[{}]", func.qualname, b, lo), func.blocks[b].insts.get(lo))
                } else {
                    (format!("{} (block {})[{}:{}]", func.qualname, b, lo, hi), None)
                }
            };
            for (b, version) in versions {
                for site in &version.graphs {
                    if shown.contains(&Arc::as_ptr(site)) || site.outer().is_some() {
                        continue;
                    }
                    shown.push(Arc::as_ptr(site));
                    if site.entry && site.short().is_some() {
                        continue;
                    }
                    if let Some(count) = site.short() {
                        // A run that is too short to be a graph, and that
                        // no path from it made longer.
                        let seed = site.seed.borrow().clone();
                        if seed.root.0 == seed.land.0 && holds(b, seed.root.1, seed.land.1) {
                            continue;
                        }
                        let (place, inst) = place(b, seed.root.1, seed.land.1);
                        let reason = format!("{} operations, fewer than min_ops ({})", count, site.min_ops);
                        left(py, place, reason, inst.map(|inst| func.op_repr(&inst.op)))?;
                        continue;
                    }
                    let path = site.path();
                    let paths = site.paths();
                    let (compiled, _) = site.forms();
                    let forms: Vec<(Arc<crate::graph::Path>, Arc<crate::graph::Exported>)> = paths
                        .iter()
                        .flat_map(|path| path.exported.borrow().iter().map(|form| (path.clone(), form.clone())).collect::<Vec<_>>())
                        .collect();
                    let closed = forms.iter().find_map(|(_, form)| form.closed.clone());
                    let (state, reason) = match &*site.state.borrow() {
                        GraphState::GivenUp(reason) if compiled > 0 => ("compiled", Some(reason.clone())),
                        GraphState::GivenUp(reason) => ("vm", Some(reason.clone())),
                        GraphState::Waiting if compiled > 0 => ("compiled", None),
                        GraphState::Waiting if closed.is_some() => ("vm", closed),
                        GraphState::Waiting => ("waiting", None),
                        GraphState::Built | GraphState::Small(_) => ("built", None),
                    };
                    let d = PyDict::new(py);
                    // A graph is named by its first operation, whether its
                    // path starts there or with the function.
                    let named = match path.first {
                        Some((block, first)) => format!("{} (block {})[{}]", func.qualname, block, first),
                        None => site.name.clone(),
                    };
                    d.set_item("where", named)?;
                    d.set_item("state", state)?;
                    if let Some((caller, graph)) = Self::holder(&held, cf, site) {
                        // Its operations are those of a graph of a caller,
                        // which is the one that runs them.
                        let name = match graph.path().first {
                            Some((block, first)) => format!("{} (block {})[{}]", caller.func.qualname, block, first),
                            None => graph.name.clone(),
                        };
                        d.set_item("within", name)?;
                    }
                    if let Some(reason) = reason {
                        d.set_item("reason", reason)?;
                    }
                    d.set_item("ops", path.operators.len())?;
                    d.set_item("operators", path.operators.clone())?;
                    d.set_item("inputs", path.inputs.len())?;
                    d.set_item("outputs", path.outputs.len())?;
                    d.set_item("branches", path.branches)?;
                    d.set_item("calls", path.calls)?;
                    d.set_item("paths", paths.len())?;
                    let mut ends: Vec<(usize, usize)> = paths.iter().map(|path| path.land).collect();
                    if ends.is_empty() {
                        ends.push(path.land);
                    }
                    ends.dedup();
                    d.set_item("ends", ends)?;
                    d.set_item("why", path.ends.clone())?;
                    d.set_item("variants", compiled)?;
                    d.set_item("hits", site.hits.load(Ordering::Relaxed))?;
                    d.set_item("misses", site.misses.load(Ordering::Relaxed))?;
                    d.set_item("exports", site.exports.load(Ordering::Relaxed))?;
                    d.set_item("version", version.id)?;
                    let listed = PyList::empty(py);
                    for (path, form) in &forms {
                        let one = PyDict::new(py);
                        one.set_item("checks", unsafe { form.conditions(path) })?;
                        one.set_item("hits", form.hits.load(Ordering::Relaxed))?;
                        if let Some(reason) = &form.closed {
                            one.set_item("reason", reason.clone())?;
                        }
                        listed.append(one)?;
                    }
                    d.set_item("forms", listed)?;
                    out.append(d)?;
                }
                for entry in &version.left {
                    // What a path of a graph of the block goes through, or
                    // takes as an operation, is not left out.
                    let taken = version.graphs.iter().any(|site| {
                        site.paths().iter().any(|path| {
                            let behind = path.land.0 != b || (entry.lo as usize) < path.land.1;
                            path.root.0 == b && path.root.1 <= entry.lo as usize && behind && path.operators.len() >= site.min_ops
                        })
                    });
                    if taken || holds(b, entry.lo as usize, entry.hi as usize) {
                        continue;
                    }
                    let (place, inst) = place(b, entry.lo as usize, entry.hi as usize);
                    left(py, place, entry.reason.clone(), inst.map(|inst| func.op_repr(&inst.op)))?;
                }
            }
        }
        Ok(out.into_any().unbind())
    }
}

impl CfgFunction {
    /// The operations that the paths of graphs hold, each by its function
    /// and its place in the code of it, with the graph and the function it
    /// is a graph of: also the operations of functions a path went into.
    fn held(&self) -> Vec<(*const Compiled, u32, Arc<Compiled>, Arc<GraphSite>)> {
        let mut held = Vec::new();
        for (cf, _, _, site) in self.graph_sites() {
            if matches!(*site.state.borrow(), GraphState::GivenUp(_) | GraphState::Small(_)) || site.outer().is_some() {
                continue;
            }
            for path in site.paths().iter().filter(|path| path.operators.len() >= site.min_ops) {
                let nodes = path.body.iter().filter(|inst| inst.kind == crate::graph::Kind::Node);
                held.extend(nodes.map(|inst| (path.owners[inst.owner as usize], inst.bc, cf.clone(), site.clone())));
            }
        }
        held
    }

    /// The graph of a caller, with its function, whose path holds the first
    /// operation of a graph of a function it went into: where that one
    /// runs, this one's operations are its own.
    fn holder(
        held: &[(*const Compiled, u32, Arc<Compiled>, Arc<GraphSite>)],
        cf: &Arc<Compiled>,
        site: &Arc<GraphSite>,
    ) -> Option<(Arc<Compiled>, Arc<GraphSite>)> {
        // The first operation of this graph, which may be one of a
        // function it goes into itself, is on a path of a graph of another
        // function that goes through this one.
        let path = site.path();
        let node = path.body.iter().find(|inst| inst.kind == crate::graph::Kind::Node)?;
        let first = (path.owners[node.owner as usize], node.bc);
        let of_caller = |(owner, at, caller, graph): &&(*const Compiled, u32, Arc<Compiled>, Arc<GraphSite>)| {
            (*owner, *at) == first
                && !Arc::ptr_eq(caller, cf)
                && graph.paths().iter().any(|path| path.owners.contains(&Arc::as_ptr(cf)))
        };
        held.iter().find(of_caller).map(|(_, _, caller, graph)| (caller.clone(), graph.clone()))
    }

    /// The graphs of the live block versions of the function and of the
    /// functions it called, each with its function, block and version.
    fn graph_sites(&self) -> Vec<(Arc<Compiled>, usize, u32, Arc<GraphSite>)> {
        let mut found = Vec::new();
        let mut functions = vec![self.inner.clone()];
        functions.extend(self.inner.callees.compiled());
        for cf in functions {
            let state = cf.state.borrow();
            for (b, bs) in state.blocks.iter().enumerate() {
                for version in &bs.versions {
                    for site in &version.graphs {
                        found.push((cf.clone(), b, version.id, site.clone()));
                    }
                }
            }
        }
        found
    }
}

/// The facts of an instruction as `CfgFunction.facts` gives them.
fn facts_dict<'py>(py: Python<'py>, facts: &InstFacts) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("runs", facts.runs)?;
    let kernels = match facts.kernels {
        Kernels::Unseen => None,
        Kernels::None => Some("none"),
        Kernels::One => Some("one"),
        Kernels::Many => Some("many"),
    };
    d.set_item("kernels", kernels)?;
    d.set_item("frames", facts.frames)?;
    d.set_item("raised", facts.raised)?;
    match &facts.callee {
        CalleeFact::Unseen => d.set_item("callee", py.None())?,
        CalleeFact::Many => d.set_item("callee", "many")?,
        CalleeFact::One(code, kind) => {
            let code = unsafe { Bound::from_borrowed_ptr(py, code.as_ptr()) };
            let callee = PyDict::new(py);
            callee.set_item("name", code.getattr("co_qualname")?)?;
            let kind = match kind {
                CalleeKind::Function => "function",
                CalleeKind::Method => "method",
                CalleeKind::Instance => "instance",
            };
            callee.set_item("kind", kind)?;
            callee.set_item("code", code)?;
            d.set_item("callee", callee)?;
        }
    }
    d.set_item("wrote", facts.wrote)?;
    d.set_item("operators", facts.operators.clone())?;
    Ok(d)
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

    /// The instructions as dicts with keys index, op, dst, uses and text;
    /// replayable; kernel, the operator an instruction was
    /// seen to be exactly one call of (`aten::mul.Tensor`), or None; and
    /// facts, what the profiling tier saw it do (see `CfgFunction.facts`),
    /// or None for one that did not run there.
    fn instructions(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let (unit, cx) = self.parts()?;
        let func = cx.func();
        let out = PyList::empty(py);
        for (i, u) in unit.insts.iter().enumerate() {
            let d = PyDict::new(py);
            let text = func.op_repr(&u.inst.op);
            d.set_item("index", i)?;
            d.set_item("op", text.split_whitespace().next().unwrap_or(""))?;
            d.set_item("dst", if u.inst.dst == NONE { None } else { Some(u.inst.dst) })?;
            let mut uses = Vec::new();
            u.inst.op.for_each_use(|v| uses.push(v));
            d.set_item("uses", uses)?;
            d.set_item("text", text)?;
            d.set_item("replayable", u.origin as usize == i && u.inst.op.is_replayable())?;
            let slot = cx.slots.get(u.origin as usize).and_then(|slot| slot.as_ref());
            d.set_item("kernel", slot.and_then(|slot| unsafe { slot.operator() }))?;
            match cx.facts.get(u.origin as usize).filter(|facts| facts.runs > 0) {
                Some(facts) => d.set_item("facts", facts_dict(py, facts)?)?,
                None => d.set_item("facts", py.None())?,
            }
            out.append(d)?;
        }
        Ok(out.into_any().unbind())
    }

    /// What the profiling tier saw the SSA value be, or None.
    fn observed(&self, py: Python<'_>, value: V) -> PyResult<Py<PyAny>> {
        let (_, cx) = self.parts()?;
        let Some(obs) = cx.obs.get(value as usize) else {
            return Err(PyIndexError::new_err("no such value"));
        };
        let d = PyDict::new(py);
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
            ValueObs::Float => d.set_item("kind", "float")?,
            ValueObs::None => d.set_item("kind", "none")?,
            ValueObs::Str { same, .. } => {
                d.set_item("kind", "str")?;
                d.set_item("same", *same)?;
            }
            ValueObs::Object { len, .. } => {
                d.set_item("kind", "object")?;
                d.set_item("len", *len)?;
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
    fn note(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let (unit, cx) = self.parts()?;
        let state = cx.cf.state.borrow();
        Ok(match &state.blocks[unit.block].note {
            Some(note) => unsafe { Bound::from_borrowed_ptr(py, note.as_ptr()).unbind() },
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
        // The variables the range binds are bound once it is done: to what
        // they were bound to last, if that is from before the range or one
        // of the values it gives, and to nothing otherwise.
        let mut bound: Vec<(u32, V)> = Vec::new();
        for u in &unit.insts[start..end] {
            if let Op::Hold(values) = &u.inst.op {
                for v in values {
                    if (!defined.contains(v) || outputs.contains(v)) && !held.contains(v) {
                        held.push(*v);
                    }
                }
            }
            if let Op::StoreLocal(slot, v) = &u.inst.op {
                bound.retain(|(other, _)| other != slot);
                let given = !defined.contains(v) || outputs.contains(v);
                bound.push((*slot, if given { *v } else { UNDEF }));
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
        for (slot, v) in bound {
            fused.push(crate::spec::UInst {
                inst: Inst { dst: NONE, op: Op::StoreLocal(slot, v), bc, kills: Vec::new() },
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
        } else if let Ok(b) = value.cast::<pyo3::types::PyBool>() {
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

/// What a builtin that goes over an object is handed in place of the object
/// where a path has called the `__iter__` of the object already: asked for
/// an iterator it gives what that call returned, as it is, so the builtin
/// does with it what it does with what an `__iter__` returns (it raises
/// for what is no iterator, and does not ask the iterator for one again).
#[pyclass(module = "torch._cfg_rs", name = "Iterated", frozen)]
pub struct Iterated {
    pub returned: Py<PyAny>,
}

#[pymethods]
impl Iterated {
    fn __iter__(&self, py: Python<'_>) -> Py<PyAny> {
        self.returned.clone_ref(py)
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
                let _ = err.value(py).setattr("cfg_index", index);
                Err(err)
            }
        }
    }

    // What follows is about the function as a whole, for code that goes
    // on from the range into other blocks.

    /// The values that are needed from instruction `index` of a block on;
    /// with `read`, only those that an instruction reads, not the ones that
    /// are merely kept alive to the end of their scope.
    #[pyo3(signature = (block, index, read=false))]
    fn live(&self, block: usize, index: usize, read: bool) -> PyResult<Vec<V>> {
        let compiled = self.compiled()?;
        let fits = compiled.func.blocks.get(block).is_some_and(|found| index <= found.insts.len());
        if !fits {
            return Err(PyIndexError::new_err("no such instruction"));
        }
        if read {
            return Ok(compiled.read_at(block, index).iter().collect());
        }
        Ok(compiled.live_at(block, index).iter().collect())
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
/// Says that what is running is not yet what it will be: a graph is
/// still running its instructions, and what a pass makes of them will run
/// in their place. A measurement that this falls into does not count
/// against the compiled form of a function.
#[pyfunction]
pub fn unsettled() {
    vm::unsettled();
}

/// Has the measurement of a function against the interpreter wait, with
/// `settle()`, for the devices that run what it started: see
/// `TRIAL_SETTLE`. `settle` must be a builtin, which runs without a frame
/// for the hook to be offered; None for no waiting.
#[pyfunction]
pub fn trial_settles(settle: &Bound<'_, PyAny>) -> PyResult<()> {
    if settle.is_none() {
        vm::trial_settles(null_mut());
    } else if unsafe { ffi::PyCFunction_Check(settle.as_ptr()) } == 0 {
        return Err(pyo3::exceptions::PyTypeError::new_err("trial_settles takes a builtin"));
    } else {
        vm::trial_settles(settle.as_ptr());
    }
    Ok(())
}

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
    Python::attach(|py| {
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

// ----------------------------------------------------------------------
// Passes written in Rust, in a library of the user's (see the crate in
// ../sdk, which is the other side of what follows)
// ----------------------------------------------------------------------

const NATIVE_ABI_VERSION: u32 = 1;

type NativeText = unsafe extern "C" fn(*mut c_void, usize, *mut u8, usize) -> usize;
type NativeRun = unsafe extern "C" fn(*mut c_void, *mut c_void, *const NativeApi, *mut c_void) -> i32;

#[repr(C)]
pub struct NativeApi {
    version: u32,
    len: unsafe extern "C" fn(*mut c_void) -> usize,
    op: NativeText,
    kernel: NativeText,
    dst: unsafe extern "C" fn(*mut c_void, usize) -> u32,
    uses: unsafe extern "C" fn(*mut c_void, usize, *mut u32, usize) -> usize,
    replayable: unsafe extern "C" fn(*mut c_void, usize) -> i32,
    fuse: unsafe extern "C" fn(*mut c_void, usize, usize, *mut c_void) -> i32,
    swap: unsafe extern "C" fn(*mut c_void, usize, usize, *mut c_void, u32) -> i32,
    error: unsafe extern "C" fn(*const c_char),
}

#[repr(C)]
struct NativeRaw {
    name: *const c_char,
    run: NativeRun,
    state: *mut c_void,
}

/// The region a native pass was called with: the Python object, which the
/// functions below reach the block through.
unsafe fn native_region<'py>(region: *mut c_void) -> Bound<'py, Region> {
    let py = Python::assume_attached();
    Bound::from_borrowed_ptr(py, region as Obj).cast_into_unchecked()
}

unsafe fn native_inst<T>(region: *mut c_void, index: usize, read: impl FnOnce(&spec::UInst, &PassCx) -> T) -> Option<T> {
    let region = native_region(region);
    let region = region.borrow();
    let (unit, cx) = region.parts().ok()?;
    unit.insts.get(index).map(|inst| read(inst, cx))
}

unsafe fn native_text(text: Option<String>, buf: *mut u8, cap: usize) -> usize {
    let text = text.unwrap_or_default();
    std::ptr::copy_nonoverlapping(text.as_ptr(), buf, text.len().min(cap));
    text.len()
}

unsafe extern "C" fn native_len(region: *mut c_void) -> usize {
    let region = native_region(region);
    let region = region.borrow();
    region.parts().map_or(0, |(unit, _)| unit.insts.len())
}

unsafe extern "C" fn native_op(region: *mut c_void, index: usize, buf: *mut u8, cap: usize) -> usize {
    let name = native_inst(region, index, |u, cx| {
        cx.func().op_repr(&u.inst.op).split_whitespace().next().unwrap_or("").to_string()
    });
    native_text(name, buf, cap)
}

unsafe extern "C" fn native_kernel(region: *mut c_void, index: usize, buf: *mut u8, cap: usize) -> usize {
    let name = native_inst(region, index, |u, cx| {
        cx.slots.get(u.origin as usize).and_then(|slot| slot.as_ref()).and_then(|slot| slot.operator())
    });
    native_text(name.flatten(), buf, cap)
}

unsafe extern "C" fn native_dst(region: *mut c_void, index: usize) -> u32 {
    native_inst(region, index, |u, _| u.inst.dst).unwrap_or(NONE)
}

unsafe extern "C" fn native_uses(region: *mut c_void, index: usize, buf: *mut u32, cap: usize) -> usize {
    let uses = native_inst(region, index, |u, _| {
        let mut uses = Vec::new();
        u.inst.op.for_each_use(|v| uses.push(v));
        uses
    })
    .unwrap_or_default();
    std::ptr::copy_nonoverlapping(uses.as_ptr(), buf, uses.len().min(cap));
    uses.len()
}

unsafe extern "C" fn native_replayable(region: *mut c_void, index: usize) -> i32 {
    native_inst(region, index, |u, _| u.origin as usize == index && u.inst.op.is_replayable()).unwrap_or(false) as i32
}

unsafe fn native_status(py: Python<'_>, result: PyResult<bool>) -> i32 {
    match result {
        Ok(changed) => changed as i32,
        Err(err) => {
            err.restore(py);
            -1
        }
    }
}

unsafe extern "C" fn native_fuse(region: *mut c_void, start: usize, end: usize, callable: *mut c_void) -> i32 {
    let region = native_region(region);
    let py = region.py();
    let callable = Bound::from_borrowed_ptr(py, callable as Obj);
    let result = region.borrow().fuse(start, end, &callable);
    native_status(py, result)
}

unsafe extern "C" fn native_swap(
    region: *mut c_void,
    start: usize,
    end: usize,
    kernel: *mut c_void,
    checks: u32,
) -> i32 {
    let region = native_region(region);
    let py = region.py();
    let kernel = Bound::from_borrowed_ptr(py, kernel as Obj);
    let result = py
        .import("torch.vm._passes")
        .and_then(|module| module.getattr("fuse_checked"))
        .and_then(|fuse| fuse.call1((&region, start, end, kernel, checks)))
        .and_then(|done| done.extract::<bool>());
    native_status(py, result)
}

unsafe extern "C" fn native_error(message: *const c_char) {
    ffi::PyErr_SetString(ffi::PyExc_RuntimeError, message);
}

static NATIVE_API: NativeApi = NativeApi {
    version: NATIVE_ABI_VERSION,
    len: native_len,
    op: native_op,
    kernel: native_kernel,
    dst: native_dst,
    uses: native_uses,
    replayable: native_replayable,
    fuse: native_fuse,
    swap: native_swap,
    error: native_error,
};

/// A pass of a library that `load_passes` loaded. It is an entry for
/// `passes=` as a Python pass is: called with a region, it runs.
#[pyclass(module = "torch._cfg_rs", name = "NativePass", frozen)]
pub struct NativePass {
    name: String,
    library: String,
    run: NativeRun,
    state: usize,
    bound: Option<Py<PyAny>>,
}

#[pymethods]
impl NativePass {
    #[getter]
    fn __name__(&self) -> String {
        self.name.clone()
    }

    fn __repr__(&self) -> String {
        format!("<torch.vm pass {} of {}>", self.name, self.library)
    }

    /// The pass with `object` handed to it each time it runs: the kernel it
    /// puts in, as a rule.
    fn bind(&self, object: Py<PyAny>) -> NativePass {
        NativePass {
            name: self.name.clone(),
            library: self.library.clone(),
            run: self.run,
            state: self.state,
            bound: Some(object),
        }
    }

    fn __call__(&self, py: Python<'_>, region: &Bound<'_, Region>) -> PyResult<bool> {
        let bound = self.bound.as_ref().map_or(null_mut(), |b| b.as_ptr());
        let status = unsafe {
            (self.run)(self.state as *mut c_void, region.as_ptr() as *mut c_void, &NATIVE_API, bound as *mut c_void)
        };
        match status {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(PyErr::take(py)
                .unwrap_or_else(|| PyRuntimeError::new_err(format!("the pass {} failed", self.name)))),
        }
    }
}

#[cfg(unix)]
extern "C" {
    fn dlopen(filename: *const c_char, flag: std::os::raw::c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}

/// Loads a library of passes (a `cdylib` built with the crate in
/// torch/csrc/cfg/rust/sdk) and returns its passes by name. The library
/// stays loaded.
#[pyfunction]
pub fn load_passes(py: Python<'_>, path: &str) -> PyResult<Py<PyAny>> {
    #[cfg(not(unix))]
    {
        let _ = (py, path);
        return Err(PyRuntimeError::new_err("loading passes is not supported on this platform"));
    }
    #[cfg(unix)]
    unsafe {
        let name = std::ffi::CString::new(path).map_err(|_| PyValueError::new_err("a path with a NUL in it"))?;
        // RTLD_NOW: a library that does not resolve fails here, not in a pass.
        let handle = dlopen(name.as_ptr(), 2);
        if handle.is_null() {
            let why = dlerror();
            let why = if why.is_null() { String::new() } else { std::ffi::CStr::from_ptr(why).to_string_lossy().into_owned() };
            return Err(PyRuntimeError::new_err(format!("cannot load {path}: {why}")));
        }
        let abi = dlsym(handle, c"torch_vm_abi".as_ptr());
        let table = dlsym(handle, c"torch_vm_passes".as_ptr());
        if abi.is_null() || table.is_null() {
            return Err(PyRuntimeError::new_err(format!(
                "{path} exports no passes: it needs export_passes! of the torch-vm-pass crate"
            )));
        }
        let abi: unsafe extern "C" fn() -> u32 = std::mem::transmute(abi);
        if abi() != NATIVE_ABI_VERSION {
            return Err(PyRuntimeError::new_err(format!(
                "{path} was built for version {} of the pass interface; this runtime has version {}",
                abi(),
                NATIVE_ABI_VERSION
            )));
        }
        let table: unsafe extern "C" fn(*mut *const NativeRaw) -> usize = std::mem::transmute(table);
        let mut first: *const NativeRaw = std::ptr::null();
        let count = table(&mut first);
        let out = PyDict::new(py);
        for raw in std::slice::from_raw_parts(first, count) {
            let name = std::ffi::CStr::from_ptr(raw.name).to_string_lossy().into_owned();
            let pass = NativePass {
                name: name.clone(),
                library: path.to_string(),
                run: raw.run,
                state: raw.state as usize,
                bound: None,
            };
            out.set_item(name, Py::new(py, pass)?)?;
        }
        Ok(out.into_any().unbind())
    }
}
