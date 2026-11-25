// CFG executor: a PEP 523 eval-frame hook that runs each Python frame by
// converting its code object to a CFG (see cfg/) and interpreting that, instead
// of CPython's bytecode loop. Anything it can't run faithfully fails hard with
// an exception rather than falling back to CPython.

pub mod autograd;
pub mod effects;
pub mod ffi;
pub mod graph;
pub mod joint;
mod live;
pub mod reads;
pub mod region;
mod run;
mod spans;
pub mod torch_ops;
mod trampoline;
pub mod types;

use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::os::raw::{c_int, c_void};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::cfg::ir::{Annotations, BlockId, Cfg, Check, Exit, Lit, Op, Site, TensorSpec, Terminator, Training, ValueId};
use crate::cfg::opcodes::{self, Out};
use crate::cfg::outline::Outline;

use ffi::*;

// Threads running the executor (ACTIVE > 0, not inside outside_executor).
// The eval-frame hook is installed only while there are any: CPython doesn't
// inline Python-to-Python calls while one is, which slows the code it runs.
static INSTALLED: Mutex<usize> = Mutex::new(0);
static PREV_EVAL_FRAME: AtomicUsize = AtomicUsize::new(0);

fn live(enter: bool) {
    let mut n = INSTALLED.lock().unwrap();
    unsafe {
        let interp = pyo3::ffi::PyInterpreterState_Get();
        if enter {
            if *n == 0 {
                PREV_EVAL_FRAME.store(_PyInterpreterState_GetEvalFrameFunc(interp) as usize, Ordering::Release);
                _PyInterpreterState_SetEvalFrameFunc(interp, eval_frame);
            }
            *n += 1;
        } else {
            *n -= 1;
            if *n == 0 {
                _PyInterpreterState_SetEvalFrameFunc(interp, std::mem::transmute::<usize, EvalFrameFn>(PREV_EVAL_FRAME.load(Ordering::Acquire)));
            }
        }
    }
}

thread_local! {
    static ACTIVE: Cell<usize> = const { Cell::new(0) };
    // Set around an instr that has a clone (see Prepared::clones): (code,
    // calling frame, Compiled). The frame the instr enters for that code,
    // called from that frame, runs the clone; taken by the first match.
    pub(crate) static NEXT_CLONE: Cell<(usize, usize, usize)> = const { Cell::new((0, 0, 0)) };
}

thread_local! {
    // Why the last elision attempt was refused (see TORCH_RUST_ELIDE_DEBUG).
    static ELIDE_WHY: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

fn why(s: String) {
    ELIDE_WHY.set(Some(s));
}

// How deep clones nest (a clone's instrs get clones of their own).
const MAX_CLONE_DEPTH: u32 = 32;

// A copy of a callee's Compiled that only frames entered from one site run,
// so its observations are that site's (see Prepared::clones).
pub struct SiteClone {
    // Held so that the address the hand-off compares can't be reused.
    pub code: Py<PyAny>,
    pub compiled: Arc<Compiled>,
}

// Runs f with no executor active on this thread: frames it runs are left to
// CPython, and no effects are recorded.
pub(crate) fn outside_executor<R>(f: impl FnOnce() -> R) -> R {
    struct Restore(usize, *mut effects::Recorder);
    impl Drop for Restore {
        fn drop(&mut self) {
            ACTIVE.set(self.0);
            effects::set_current(self.1);
        }
    }
    struct Live;
    impl Drop for Live {
        fn drop(&mut self) {
            live(true);
        }
    }
    let _restore = Restore(ACTIVE.replace(0), effects::set_current(std::ptr::null_mut()));
    let _live = (_restore.0 > 0).then(|| {
        live(false);
        Live
    });
    f()
}

const CO_GENERATOR: u32 = 0x20;
const CO_COROUTINE: u32 = 0x80;
const CO_ITERABLE_COROUTINE: u32 = 0x100;
const CO_ASYNC_GENERATOR: u32 = 0x200;
const GENERATOR_FLAGS: u32 = CO_GENERATOR | CO_COROUTINE | CO_ITERABLE_COROUTINE | CO_ASYNC_GENERATOR;

// Objects something else keeps alive, only touched with the GIL held.
pub struct Borrowed(pub Vec<*mut pyo3::ffi::PyObject>);
unsafe impl Send for Borrowed {}
unsafe impl Sync for Borrowed {}

// A Cfg with what the executor precomputes for it, and the same for the
// origins of its blocks.
pub struct Prepared {
    pub cfg: Arc<Cfg>,
    // For an instr with several outputs, the Project instr receiving each
    // (by output position).
    pub projects: Vec<Vec<Option<ValueId>>>,
    // For each of a block's reads (Block::reads), its position there.
    pub read_pos: Vec<u32>,
    // Number of new outputs of each Op::Bytecode instr.
    pub nouts: Vec<u32>,
    // By block, then version: the body as run_blocks steps through it.
    pub code: Vec<Vec<run::Code>>,
    // Each LoadGlobal's cache, at its index in `global_slot` (by instr).
    pub globals: Box<[ffi::GilCell<ffi::GlobalCache>]>,
    pub global_slot: Vec<u32>,
    pub live: live::Liveness,
    // By block.
    pub origins: Vec<Option<Prepared>>,
    // What has been observed running the Cfg. Shared with the Prepared of an
    // optimized copy of it, whose values keep their ids.
    pub effects: Arc<ffi::GilCell<effects::Table>>,
    // By value: the join of every object seen there. Observed once per
    // value an instr makes, so not behind a lock.
    pub types: Arc<ffi::GilCell<Vec<types::Slot>>>,
    pub callees: Arc<ffi::GilCell<effects::Callees>>,
    // For each Op::Aten instr, the operator overload's C++ binding and its
    // kwnames (None for none).
    pub aten: HashMap<ValueId, (Py<PyAny>, Option<Py<PyTuple>>)>,
    // Cfg::literals as Python objects.
    pub literals: Vec<Py<PyAny>>,
    // By block: how many times no version's guard held (see _cfg_misses).
    pub misses: Box<[AtomicU32]>,
    // By block: whether it may get its own autograd node (see autograd.rs);
    // cleared when it can't, with why.
    pub autograd: Box<[AtomicBool]>,
    pub autograd_why: Box<[ffi::GilCell<Option<String>>]>,
    // Cfg::trainings, prepared.
    pub trainings: Vec<Arc<autograd::TrainingPrep>>,
    // By block: how many times its last version ran since it was last
    // traced into a region, and whether it can't be (see region.rs).
    pub region_runs: Box<[AtomicU32]>,
    pub region_off: Box<[AtomicBool]>,
    // By block: the types of its last version's params then inputs over its
    // runs while observing (see region::observing; torch.vm's input_types).
    pub region_types: Box<[ffi::GilCell<Vec<types::Slot>>]>,
    // The Prepareds (by address) whose Cfgs this one's derives from by adding
    // regions only. A frame that runs one of them is what keeps it (see
    // Compiled::plans): an address here is only compared with a running
    // frame's.
    pub region_lineage: ffi::GilCell<Vec<usize>>,
    // Something refers to it by address for longer than a frame runs it (an
    // autograd node: see autograd.rs's Where): its Plan is never freed.
    pub kept: AtomicBool,
    // Whether running it records (see run.rs's recording): until every block
    // of the Cfg has been offered to the optimizer, which reads nothing
    // recorded after that. Optimized versions never record, and origins always
    // do, so guard misses are still seen.
    pub record: bool,
    // What each of Cfg::pins was pinned to.
    pub pins: Vec<Py<PyAny>>,
    // What Op::Const and name indices refer to: the code's co_consts and
    // co_names, then the Cfg's own (see Cfg::const_pins), borrowed from the
    // Compiled, `pins` and `extra_names`. Empty for an origin (it uses its
    // owner's).
    pub consts: Borrowed,
    pub names: Borrowed,
    pub extra_names: Vec<Py<PyAny>>,
    // By instr (a call, or an attribute load whose class runs __getattr__):
    // the clone the callee's frame runs when entered from it. Made when the
    // code is optimized, for Python functions every visit entered directly,
    // and carried over to later Plans.
    pub clones: HashMap<ValueId, SiteClone>,
}

impl Prepared {
    // `prev`: the Prepared of the Cfg this one replaces (with the same value
    // ids), whose observations carry over.
    fn new(cfg: Arc<Cfg>, prev: Option<&Prepared>) -> PyResult<Self> {
        let n = cfg.instrs.len();
        let mut projects: Vec<Vec<Option<ValueId>>> = vec![vec![]; n];
        let mut nouts = vec![0u32; n];
        let mut flags = vec![0u8; n];
        let mut read_pos = vec![u32::MAX; n];
        for b in &cfg.blocks {
            for (i, r) in b.reads.iter().enumerate() {
                read_pos[r.idx()] = i as u32;
            }
        }
        let body = || cfg.blocks.iter().flat_map(|b| b.reads.iter().chain(b.versions.iter().flat_map(|ver| &ver.body)));
        for v in body() {
            let ins = &cfg[*v];
            match &ins.op {
                Op::Project(k) => {
                    let ps = &mut projects[ins.args[0].idx()];
                    if ps.len() <= *k as usize {
                        ps.resize(*k as usize + 1, None);
                    }
                    ps[*k as usize] = Some(*v);
                }
                Op::Bytecode { opcode, arg } => {
                    let sig = opcodes::signature(*opcode, *arg, false).expect("converted op has a signature");
                    nouts[v.idx()] = sig.outputs.iter().filter(|o| **o == Out::New).count() as u32;
                }
                _ => {}
            }
            flags[v.idx()] = match ins.op {
                Op::Yield => run::FLAG_YIELD,
                Op::Call { .. } => run::FLAG_CALL,
                Op::Bytecode {
                    opcode: opcodes::CALL_FUNCTION_EX,
                    ..
                } => run::FLAG_CALL_EX,
                _ => 0,
            } | if ins.op.can_raise() { run::FLAG_RAISE } else { 0 };
        }
        // A multi-output instr's values are its Projects'.
        for v in body() {
            if cfg[*v].op.produces_value() && projects[v.idx()].is_empty() {
                flags[v.idx()] |= run::FLAG_VALUE;
            }
        }
        let mut aten = HashMap::new();
        for v in cfg.blocks.iter().flat_map(|b| &b.versions).flat_map(|ver| &ver.body) {
            if let Op::Aten { name, kwnames } = &cfg[*v].op {
                let kw = Python::attach(|py| (!kwnames.is_empty()).then(|| PyTuple::new(py, kwnames).map(|t| t.unbind())).transpose())?;
                aten.insert(*v, (resolve_aten(name)?, kw));
            }
        }
        let literals = Python::attach(|py| cfg.literals.iter().map(|l| lit_object(py, l)).collect::<PyResult<Vec<_>>>())?;
        let live = live::compute(&cfg);
        let code = run::Code::build(&cfg, &live, &flags);
        let mut global_slot = vec![u32::MAX; n];
        let mut nglobals = 0;
        for v in body() {
            if let Op::LoadGlobal(_) = cfg[*v].op {
                global_slot[v.idx()] = nglobals;
                nglobals += 1;
            }
        }
        let globals = (0..nglobals).map(|_| ffi::GilCell::new(ffi::GlobalCache::default())).collect();
        if let Some(y) = live.bad_yield {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                "CFG executor: a value not on the stack is live across the yield at offset {}",
                cfg[y].origin * 2
            )));
        }
        // An origin the previous Cfg already had is the same Cfg.
        let origins = cfg
            .blocks
            .iter()
            .map(|b| {
                b.origin
                    .clone()
                    .map(|o| {
                        let same = prev.and_then(|p| p.find(&o));
                        Prepared::new(o, same)
                    })
                    .transpose()
            })
            .collect::<PyResult<_>>()?;
        let (effects, types, callees) = match prev {
            Some(p) => {
                unsafe { &mut *p.types.get() }.resize(n, types::Slot::default());
                (p.effects.clone(), p.types.clone(), p.callees.clone())
            }
            None => (
                Arc::new(ffi::GilCell::new(effects::Table::default())),
                Arc::new(ffi::GilCell::new(vec![types::Slot::default(); n])),
                Arc::new(ffi::GilCell::new(effects::Callees::default())),
            ),
        };
        let cfg_blocks = cfg.blocks.len();
        let autograd = (0..cfg_blocks).map(|b| AtomicBool::new(autograd::ineligible(&cfg, BlockId(b as u32)).is_none())).collect();
        let trainings = cfg
            .trainings
            .iter()
            .map(|t| match prev.and_then(|p| p.trainings.iter().find(|q| Arc::ptr_eq(&q.training, t))) {
                Some(q) => Ok(q.clone()),
                None => autograd::TrainingPrep::new(t.clone()).map(Arc::new),
            })
            .collect::<PyResult<_>>()?;
        Ok(Prepared {
            cfg,
            projects,
            read_pos,
            nouts,
            code,
            globals,
            global_slot,
            live,
            origins,
            effects,
            types,
            callees,
            aten,
            literals,
            misses: (0..cfg_blocks).map(|_| AtomicU32::new(0)).collect(),
            autograd,
            autograd_why: (0..cfg_blocks).map(|_| ffi::GilCell::new(None)).collect(),
            trainings,
            region_runs: (0..cfg_blocks).map(|_| AtomicU32::new(0)).collect(),
            region_off: (0..cfg_blocks).map(|_| AtomicBool::new(false)).collect(),
            region_types: (0..cfg_blocks).map(|_| ffi::GilCell::new(vec![])).collect(),
            region_lineage: ffi::GilCell::new(vec![]),
            kept: AtomicBool::new(false),
            record: true,
            // The pins the previous Cfg already had (pins are only added).
            pins: Python::attach(|py| prev.map_or(vec![], |p| p.pins.iter().map(|o| o.clone_ref(py)).collect())),
            clones: HashMap::new(),
            consts: Borrowed(vec![]),
            names: Borrowed(vec![]),
            extra_names: vec![],
        })
    }

    // The Prepared of `cfg` here or among the origins.
    fn find(&self, cfg: &Arc<Cfg>) -> Option<&Prepared> {
        if Arc::ptr_eq(&self.cfg, cfg) {
            return Some(self);
        }
        self.origins.iter().flatten().find_map(|o| o.find(cfg))
    }
}

pub(crate) fn lit_object(py: Python<'_>, l: &Lit) -> PyResult<Py<PyAny>> {
    let torch = || py.import("torch");
    Ok(match l {
        Lit::None => py.None(),
        Lit::Bool(b) => pyo3::types::PyBool::new(py, *b).to_owned().into_any().unbind(),
        Lit::Int(i) => i.into_pyobject(py)?.into_any().unbind(),
        Lit::Float(bits) => f64::from_bits(*bits).into_pyobject(py)?.into_any().unbind(),
        Lit::Str(x) => x.into_pyobject(py)?.into_any().unbind(),
        Lit::IntList(xs) => pyo3::types::PyList::new(py, xs)?.into_any().unbind(),
        Lit::FloatList(xs) => pyo3::types::PyList::new(py, xs.iter().map(|b| f64::from_bits(*b)))?.into_any().unbind(),
        Lit::BoolList(xs) => pyo3::types::PyList::new(py, xs)?.into_any().unbind(),
        Lit::Dtype(x) | Lit::Layout(x) | Lit::MemoryFormat(x) => torch()?.getattr(x.as_str())?.unbind(),
        Lit::Device(x) => torch()?.getattr("device")?.call1((x.as_str(),))?.unbind(),
    })
}

// torch.ops.<ns>.<op>.<overload>._op for "ns::op.overload" (or "ns::op").
pub(crate) fn resolve_aten(name: &str) -> PyResult<Py<PyAny>> {
    let err = || pyo3::exceptions::PyValueError::new_err(format!("not an operator name: {name}"));
    let (ns, rest) = name.split_once("::").ok_or_else(err)?;
    let (op, overload) = rest.split_once('.').unwrap_or((rest, "default"));
    Python::attach(|py| {
        let ops = py.import("torch")?.getattr("ops")?;
        Ok(ops.getattr(ns)?.getattr(op)?.getattr(overload)?.getattr("_op")?.unbind())
    })
}

// The classes whose Python operators run as plain aten ops: torch.Tensor and
// nn.Parameter (whose __torch_function__ is disabled).
fn plain_tensor_types() -> [usize; 2] {
    static TYPES: OnceLock<[usize; 2]> = OnceLock::new();
    *TYPES.get_or_init(|| {
        Python::attach(|py| {
            let get = |m: &str, n: &str| py.import(m).and_then(|m| m.getattr(n)).map_or(0, |t| t.as_ptr() as usize);
            [get("torch", "Tensor"), get("torch.nn", "Parameter")]
        })
    })
}

pub(crate) unsafe fn is_plain_tensor(obj: *mut pyo3::ffi::PyObject) -> bool {
    plain_tensor_types().contains(&(pyo3::ffi::Py_TYPE(obj) as usize))
}

// What a frame of the code runs: the current Cfg, prepared.
pub struct Plan {
    pub prep: Prepared,
    // Where a generator frame resumes, by the code unit its prev_instr points
    // at: each Yield, and RETURN_GENERATOR for a just-created generator. The
    // instr after it is the RESUME's Op::Resume. (block, version, index).
    // Outlining and optimization keep the blocks holding them, so they are all
    // in the top-level Cfg, and a generator suspended under one Plan resumes
    // at the same place under its replacement.
    pub resume: HashMap<u32, (BlockId, usize, usize)>,
    // How many frames are running it, and how many callers are reading it
    // while they build its replacement (see Compiled::plans).
    users: AtomicU32,
}

// Keeps a Plan from being freed while it is read outside a frame.
struct Using<'a>(&'a Plan);

impl<'a> Using<'a> {
    fn new(plan: &'a Plan) -> Self {
        plan.users.fetch_add(1, Ordering::Relaxed);
        Using(plan)
    }
}

impl Drop for Using<'_> {
    fn drop(&mut self) {
        self.0.users.fetch_sub(1, Ordering::Relaxed);
    }
}

impl Plan {
    // `consts` and `names`: the code's co_consts and co_names.
    fn new(cfg: Cfg, prev: Option<&Plan>, record: bool, pins: Vec<Py<PyAny>>, consts: &Py<PyTuple>, names: &Py<PyTuple>) -> PyResult<Self> {
        let mut resume = HashMap::new();
        let bodies = cfg.block_ids().flat_map(|b| cfg[b].versions.iter().enumerate().map(move |(vi, ver)| (b, vi, ver)));
        for (b, vi, ver) in bodies {
            for (i, v) in ver.body.iter().enumerate() {
                if let Op::Yield
                | Op::Bytecode {
                    opcode: opcodes::RETURN_GENERATOR,
                    ..
                } = &cfg[*v].op
                {
                    if ver.body.get(i + 1).map(|r| &cfg[*r].op) != Some(&Op::Resume) {
                        return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                            "CFG executor: no RESUME after the resume point at offset {}",
                            cfg[*v].origin * 2
                        )));
                    }
                    resume.insert(cfg[*v].origin, (b, vi, i));
                }
            }
        }
        let mut prep = Prepared::new(Arc::new(cfg), prev.map(|p| &p.prep))?;
        prep.record = record;
        prep.pins.extend(pins);
        if prep.pins.len() != prep.cfg.pins.len() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("CFG executor: a pin has no object"));
        }
        Python::attach(|py| -> PyResult<()> {
            let (consts, names) = (consts.bind(py), names.bind(py));
            prep.consts.0 = consts.iter().map(|o| o.as_ptr()).collect();
            prep.consts.0.extend(prep.cfg.const_pins.iter().map(|p| prep.pins[*p as usize].as_ptr()));
            for n in &prep.cfg.names[names.len()..] {
                prep.extra_names.push(pyo3::types::PyString::intern(py, n).into_any().unbind());
            }
            prep.names.0 = names.iter().map(|o| o.as_ptr()).collect();
            prep.names.0.extend(prep.extra_names.iter().map(|o| o.as_ptr()));
            Ok(())
        })?;
        Ok(Plan { resume, prep, users: AtomicU32::new(0) })
    }
}

// Per-code-object data, cached in co_extra and freed with the code object.
pub struct Compiled {
    // The current Plan, the last of `plans`. Replaced when the Cfg is
    // optimized or a block gets its first region; frames already running
    // keep the Plan they started with. A Plan is freed (see Compiled::sweep)
    // once it is neither the current one nor the first (optimizing again
    // starts from that one), no frame runs it and nothing refers to it by
    // address.
    plan: AtomicPtr<Plan>,
    // Boxed so a Plan doesn't move when the Vec grows or shrinks.
    #[allow(clippy::vec_box)]
    plans: Mutex<Vec<Box<Plan>>>,
    // Whether the code was optimized after its first run (see OPTIMIZE), and
    // how many times its frames returned since.
    auto_optimized: AtomicBool,
    returns: AtomicU32,
    // Whether its frames run on CPython (see _cfg_bypass_cold), and with
    // `deep`, every frame they run too (outside the executor).
    cold: AtomicBool,
    deep: AtomicBool,
    // 0 for a code's own Compiled; one more than its caller's for a clone.
    pub depth: u32,
    pub flags: u32,
    pub consts: Py<PyTuple>,
    pub names: Py<PyTuple>,
    pub qualname: String,
    pub filename: String,
    // (start, end, line) of the code's units, from co_lines.
    pub lines: Vec<(u32, u32, i32)>,
    // The code object's address: with an instr's origin, a call site (see
    // _cfg_opaque_site).
    pub code: usize,
    // For a clone, the call sites that lead to it, outermost first: (the
    // caller's qualname, the call's line). Empty for a code's own Compiled.
    pub path: Vec<(String, i32)>,
    // The origins of the calls a cfg pass made movable when this was last
    // optimized: pure for its callers too (see pure_compiled).
    declared: Mutex<HashSet<u32>>,
}

fn code_lines(code: &Bound<'_, PyAny>) -> PyResult<Vec<(u32, u32, i32)>> {
    let mut out = vec![];
    for item in code.call_method0("co_lines")?.try_iter()? {
        let (start, end, line): (u32, u32, Option<i32>) = item?.extract()?;
        out.push((start / 2, end / 2, line.unwrap_or(0)));
    }
    Ok(out)
}

// Every Compiled made (codes' own and clones), for whole-process reports
// (see _cfg_region_stats).
static REGISTRY: Mutex<Vec<std::sync::Weak<Compiled>>> = Mutex::new(Vec::new());

pub(crate) fn register(c: &Arc<Compiled>) {
    REGISTRY.lock().unwrap().push(Arc::downgrade(c));
}

pub(crate) fn registered() -> Vec<Arc<Compiled>> {
    let mut reg = REGISTRY.lock().unwrap();
    reg.retain(|w| w.strong_count() > 0);
    reg.iter().filter_map(|w| w.upgrade()).collect()
}

impl Compiled {
    fn new(code: &Bound<'_, PyAny>) -> PyResult<Self> {
        let info = crate::cfg::python::code_info(code)?;
        let outline = *OUTLINE.lock().unwrap();
        let cfg = crate::cfg::build(info, true, outline).map_err(crate::cfg::python::to_pyerr)?;
        let consts = code.getattr("co_consts")?.cast_into::<PyTuple>()?.unbind();
        let names = code.getattr("co_names")?.cast_into::<PyTuple>()?.unbind();
        let mut plan = Box::new(Plan::new(cfg, None, true, vec![], &consts, &names)?);
        Ok(Compiled {
            plan: AtomicPtr::new(&mut *plan),
            plans: Mutex::new(vec![plan]),
            auto_optimized: AtomicBool::new(false),
            returns: AtomicU32::new(0),
            cold: AtomicBool::new(false),
            deep: AtomicBool::new(false),
            depth: 0,
            flags: code.getattr("co_flags")?.extract()?,
            consts,
            names,
            qualname: code.getattr("co_qualname")?.extract()?,
            filename: code.getattr("co_filename")?.extract()?,
            lines: code_lines(code)?,
            code: code.as_ptr() as usize,
            path: vec![],
            declared: Mutex::new(HashSet::new()),
        })
    }

    // The source line of code unit `unit` (0 if none).
    pub fn line(&self, unit: u32) -> i32 {
        self.lines.iter().find(|(s, e, _)| *s <= unit && unit < *e).map_or(0, |x| x.2)
    }

    pub fn plan(&self) -> &Plan {
        unsafe { &*self.plan.load(Ordering::Acquire) }
    }

    // Frees the Plans nothing uses any more (see `plan`).
    fn sweep(&self) {
        let mut plans = self.plans.lock().unwrap();
        let current = self.plan.load(Ordering::Acquire) as *const Plan;
        let mut first = true;
        plans.retain(|p| {
            let keep = first || std::ptr::eq(&**p, current) || p.users.load(Ordering::Relaxed) > 0 || p.prep.kept.load(Ordering::Relaxed);
            first = false;
            keep
        });
    }
}

// How the executor outlines the CFGs it builds (for testing outlining).
static OUTLINE: Mutex<Option<Outline>> = Mutex::new(None);

/// Outline chains of blocks in the CFGs built from now on: None, "merge" or
/// "origin" (see cfg/outline.rs). Code already run keeps its CFG.
#[pyfunction]
#[pyo3(signature = (mode))]
pub fn _set_cfg_outline(mode: Option<&str>) -> PyResult<()> {
    *OUTLINE.lock().unwrap() = crate::cfg::python::outline_mode(mode)?;
    Ok(())
}

// Whether optimizing elides calls whose clones' paths can be copied into the
// caller (see cfg/inline.rs). With OPTIMIZE, a code with clones is also
// optimized again (eliding) when its frames have returned ELIDE_AFTER times
// since its first optimization, enough for clones nested MAX_CLONE_DEPTH
// deep to have been made and to have observed.
static ELIDE: AtomicBool = AtomicBool::new(false);
static ELIDE_AFTER: AtomicU32 = AtomicU32::new(10);

/// How many returns after its first optimization a code with clones is
/// optimized again (with elision or regions on).
#[pyfunction]
pub fn _set_cfg_reoptimize_after(n: u32) {
    ELIDE_AFTER.store(n.max(1), Ordering::Relaxed);
}

/// Elide calls of Python functions when optimizing, from what their clones
/// observed (see cfg/inline.rs).
#[pyfunction]
pub fn _set_cfg_elide(on: bool) {
    ELIDE.store(on, Ordering::Relaxed);
}

// Whether the executor optimizes every block of a code's CFG after the first
// time a frame of it returns (for testing the optimizer).
static OPTIMIZE: AtomicBool = AtomicBool::new(false);

/// Optimize every block of each code's CFG after a frame of it first returns.
#[pyfunction]
pub fn _set_cfg_optimize(on: bool) {
    OPTIMIZE.store(on, Ordering::Relaxed);
}

// Whether the executor records side effects and value types (see
// cfg_observed). The optimizer works from them.
pub static RECORD: AtomicBool = AtomicBool::new(true);

/// Turn recording of side effects and value types on or off for executor
/// calls started from now on.
#[pyfunction]
pub fn _set_cfg_record(on: bool) {
    RECORD.store(on, Ordering::Relaxed);
}

unsafe extern "C" fn free_compiled(p: *mut c_void) {
    if !p.is_null() && p != trampoline::mark() {
        drop(Arc::from_raw(p as *const Compiled));
    }
}

// Every Compiled is held by an Arc: co_extra's, or a caller's (a clone).
pub(crate) unsafe fn shared(c: &Compiled) -> Arc<Compiled> {
    Arc::increment_strong_count(c);
    Arc::from_raw(c)
}

pub(crate) fn extra_index() -> isize {
    static INDEX: OnceLock<isize> = OnceLock::new();
    *INDEX.get_or_init(|| unsafe { PyUnstable_Eval_RequestCodeExtraIndex(free_compiled) })
}

enum Lookup<'a> {
    Compiled(&'a Compiled),
    // A trampoline (see trampoline.rs), run by CPython's evaluator.
    Trampoline,
    // A Python exception is set.
    Error,
}

// The cached Compiled for `code`, converting it on first use. The result lives
// as long as the code object.
unsafe fn compiled<'a>(code: *mut pyo3::ffi::PyObject) -> Lookup<'a> {
    let index = extra_index();
    let mut p: *mut c_void = std::ptr::null_mut();
    if index < 0 || PyUnstable_Code_GetExtra(code, index, &mut p) < 0 {
        return Lookup::Error;
    }
    if p == trampoline::mark() {
        return Lookup::Trampoline;
    }
    if !p.is_null() {
        return Lookup::Compiled(&*(p as *const Compiled));
    }
    Python::attach(|py| {
        let code_obj = Bound::from_borrowed_ptr(py, code);
        match Compiled::new(&code_obj) {
            Ok(c) => {
                let c = Arc::new(c);
                register(&c);
                let p = Arc::into_raw(c);
                if PyUnstable_Code_SetExtra(code, index, p as *mut c_void) < 0 {
                    drop(Arc::from_raw(p));
                    return Lookup::Error;
                }
                Lookup::Compiled(&*p)
            }
            Err(e) => {
                e.restore(py);
                Lookup::Error
            }
        }
    })
}

unsafe fn fail(msg: &str, exc: *mut pyo3::ffi::PyObject) {
    let msg = std::ffi::CString::new(msg).unwrap();
    pyo3::ffi::PyErr_SetString(exc, msg.as_ptr());
}

// Codes (by address) whose frames, and every frame they run, run on CPython
// (see _cfg_skip).
static SKIPPED: Mutex<Vec<usize>> = Mutex::new(Vec::new());
static SKIP_ANY: AtomicBool = AtomicBool::new(false);
// Those of SKIPPED that are the region tracer's own (its mode handlers, its
// float subclass): a trace through one of the others is refused.
static SKIPPED_OWN: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// Run `f`'s frames, and every frame they run, on CPython, like
/// `torch.compiler.disable`.
#[pyfunction]
#[pyo3(signature = (f, own=false))]
pub fn _cfg_skip(f: &Bound<'_, PyAny>, own: bool) -> PyResult<()> {
    let code = f.getattr("__code__")?;
    SKIPPED.lock().unwrap().push(code.as_ptr() as usize);
    if own {
        SKIPPED_OWN.lock().unwrap().push(code.as_ptr() as usize);
    }
    // Kept alive so the address isn't reused.
    std::mem::forget(code.unbind());
    SKIP_ANY.store(true, Ordering::Relaxed);
    Ok(())
}

// gc.callbacks (the same list object for the process's life), once looked up.
static GC_CALLBACKS: AtomicPtr<pyo3::ffi::PyObject> = AtomicPtr::new(std::ptr::null_mut());

// Whether `frame` runs one of gc.callbacks. The collector calls them from
// whatever allocation set it off, so they run outside the executor: neither
// recorded as an effect of the instr that allocated, nor traced into a region.
unsafe fn gc_callback(frame: *mut Frame) -> bool {
    let mut list = GC_CALLBACKS.load(Ordering::Relaxed);
    if list.is_null() {
        let py = Python::assume_attached();
        let Ok(l) = py.import("gc").and_then(|m| m.getattr("callbacks")) else {
            pyo3::ffi::PyErr_Clear();
            return false;
        };
        list = l.into_ptr();
        GC_CALLBACKS.store(list, Ordering::Relaxed);
    }
    let n = pyo3::ffi::PyList_GET_SIZE(list);
    if n == 0 {
        return false;
    }
    let func = tr_frame_func(frame);
    (0..n).any(|i| tr_bound_function(pyo3::ffi::PyList_GET_ITEM(list, i)) == func)
}

// How a frame of `code` runs while a region's body is traced, if `code`
// belongs to what traces rather than to the body: Some(true) on CPython with
// everything it calls (torch._ops and torch._library: the operator objects
// the dispatcher looks up from inside an operator), Some(false) on CPython
// with the frames it enters being the executor's again (torch.fx,
// torch.overrides and the region tracer: fx's stand-ins for a module's call
// and attribute lookup and the handlers of the function modes call the body
// back).
unsafe fn tracer_code(code: *mut pyo3::ffi::PyObject) -> Option<bool> {
    let py = Python::assume_attached();
    let Ok(f) = Bound::from_borrowed_ptr(py, code).getattr("co_filename").and_then(|f| f.extract::<String>()) else {
        pyo3::ffi::PyErr_Clear();
        return None;
    };
    let at = f.rfind("/torch/")?;
    let rest = &f[at + 7..];
    if rest == "_ops.py" || rest.starts_with("_library/") {
        Some(true)
    } else if rest.starts_with("fx/") || rest == "overrides.py" || rest == "_rust_inductor.py" {
        Some(false)
    } else {
        None
    }
}

unsafe extern "C" fn eval_frame(
    tstate: *mut pyo3::ffi::PyThreadState,
    frame: *mut Frame,
    throwflag: c_int,
) -> *mut pyo3::ffi::PyObject {
    let prev: EvalFrameFn = std::mem::transmute(PREV_EVAL_FRAME.load(Ordering::Acquire));
    if ACTIVE.get() == 0 {
        return prev(tstate, frame, throwflag);
    }
    // Resuming a generator (created by either engine): the frame is owned by
    // the generator object, and we continue from where it was suspended.
    let gen_frame = tr_frame_owned_by_generator(frame) != 0;
    let code = tr_frame_code(frame);
    // While a region's body is traced the executor runs every frame under
    // it, cold code too, so that what the frames read is recorded (see
    // reads.rs); it neither observes nor optimizes then.
    let tracing = region::tracing();
    if tracing {
        reads::frame_entered();
    }
    if !gen_frame && SKIP_ANY.load(Ordering::Relaxed) && SKIPPED.lock().unwrap().contains(&(code as usize)) {
        if NEXT_CLONE.get().0 == code as usize {
            NEXT_CLONE.set((0, 0, 0));
        }
        if reads::on() && !SKIPPED_OWN.lock().unwrap().contains(&(code as usize)) {
            let name = Bound::from_borrowed_ptr(Python::assume_attached(), tr_frame_qualname(frame)).to_string();
            reads::refuse(format!("it runs {name}, which the VM was told to skip"));
        }
        return outside_executor(|| prev(tstate, frame, throwflag));
    }
    if !gen_frame && gc_callback(frame) {
        return outside_executor(|| prev(tstate, frame, throwflag));
    }
    if tracing && let Some(deep) = tracer_code(code) {
        if NEXT_CLONE.get().0 == code as usize {
            NEXT_CLONE.set((0, 0, 0));
        }
        return if deep { outside_executor(|| prev(tstate, frame, throwflag)) } else { prev(tstate, frame, throwflag) };
    }
    let pending = NEXT_CLONE.get();
    let compiled = if !gen_frame && pending.2 != 0 && pending.0 == code as usize && tr_current_frame(tstate) as usize == pending.1 {
        NEXT_CLONE.set((0, 0, 0));
        Some(&*(pending.2 as *const Compiled))
    } else {
        match compiled(code) {
            Lookup::Compiled(c) => Some(c),
            Lookup::Trampoline => return prev(tstate, frame, throwflag),
            Lookup::Error => None,
        }
    };
    // A cold generator's frame resumes on CPython too (the executor suspends
    // frames as CPython does).
    if let Some(c) = compiled {
        // Once cold code is bypassed, code first run after (a bypassed
        // frame's callee, run by its own Compiled) is cold too.
        if !c.cold.load(Ordering::Relaxed) && BYPASSED.load(Ordering::Relaxed) && !c.auto_optimized.load(Ordering::Relaxed) {
            c.cold.store(true, Ordering::Relaxed);
        }
        if c.cold.load(Ordering::Relaxed) && !tracing {
            return if c.deep.load(Ordering::Relaxed) { outside_executor(|| prev(tstate, frame, throwflag)) } else { prev(tstate, frame, throwflag) };
        }
    }
    effects::frame_entered(frame, gen_frame);
    if !gen_frame {
        let Some(c) = compiled else {
            tr_frame_clear_and_pop(tstate, frame);
            return std::ptr::null_mut();
        };
        if c.flags & GENERATOR_FLAGS != 0 {
            // Calling a generator function runs only its prologue, up to
            // RETURN_GENERATOR, which creates the generator object (from
            // CPython internals that aren't exported). The executor runs the
            // body when the generator is resumed.
            return prev(tstate, frame, throwflag);
        }
    }
    if tr_frame_enter(tstate, frame) < 0 || compiled.is_none() {
        return leave(tstate, frame, gen_frame, run::Outcome::Error);
    }
    let c = compiled.unwrap();
    if tr_tracing_active(tstate) != 0 {
        fail(
            "CFG executor: sys.settrace/setprofile/monitoring are not supported",
            pyo3::ffi::PyExc_RuntimeError,
        );
        return leave(tstate, frame, gen_frame, run::Outcome::Error);
    }
    let plan = c.plan();
    let start = if gen_frame {
        let unit = tr_frame_get_instr(frame) as u32;
        let Some(&(b, ver, i)) = plan.resume.get(&unit) else {
            fail(
                &format!("CFG executor: can't resume a generator at offset {}", unit * 2),
                pyo3::ffi::PyExc_RuntimeError,
            );
            return leave(tstate, frame, gen_frame, run::Outcome::Error);
        };
        run::Start::Resume { b, ver, i, throw: throwflag != 0 }
    } else {
        run::Start::Entry
    };
    let in_instr = tracing && reads::in_instr();
    let outcome = {
        let _using = Using::new(plan);
        run::run(c, &plan.prep, frame, start)
    };
    // The plan was replaced while the frame ran it (the function was
    // optimized, or a block of it got its first region): it goes with the
    // last frame that runs it.
    if !std::ptr::eq(plan, c.plan()) {
        c.sweep();
    }
    if matches!(outcome, run::Outcome::Return(_)) && OPTIMIZE.load(Ordering::Relaxed) && !OPTIMIZING.get() && !tracing {
        let py = Python::assume_attached();
        if !c.auto_optimized.swap(true, Ordering::Relaxed) {
            if let Err(e) = optimize(py, c, None) {
                e.write_unraisable(py, None);
            }
            c.sweep();
        } else if (ELIDE.load(Ordering::Relaxed) || region::enabled() || PURE_CALLS.load(Ordering::Relaxed))
            && c.returns.fetch_add(1, Ordering::Relaxed) + 1 == ELIDE_AFTER.load(Ordering::Relaxed)
            && !c.plan().prep.clones.is_empty()
        {
            // By now its clones (and theirs) have been optimized and have
            // observed: optimize again, eliding what they allow.
            if let Err(e) = outside_executor(|| optimize_now(py, c, None, true)) {
                e.write_unraisable(py, None);
            }
            c.sweep();
        }
    }
    // What the frame held is dropped with no instr of the body running
    // (see reads.rs); then the caller's instr goes on.
    let result = leave(tstate, frame, gen_frame, outcome);
    if tracing {
        reads::set_in_instr(in_instr);
    }
    result
}

unsafe fn leave(
    tstate: *mut pyo3::ffi::PyThreadState,
    frame: *mut Frame,
    gen_frame: bool,
    outcome: run::Outcome,
) -> *mut pyo3::ffi::PyObject {
    let result = match outcome {
        run::Outcome::Yield { value, unit } => {
            tr_gen_suspend(tstate, frame, unit as c_int);
            return value;
        }
        run::Outcome::Return(v) => v,
        run::Outcome::Error => std::ptr::null_mut(),
        run::Outcome::Leave(_) | run::Outcome::Outer(..) | run::Outcome::Stopped(_) => unreachable!("only an outlined Cfg leaves"),
    };
    if gen_frame {
        tr_gen_exit(tstate, frame);
    } else {
        tr_frame_exit(tstate, frame);
    }
    result
}

pub(crate) struct ExecGuard;

impl ExecGuard {
    pub(crate) fn new() -> Self {
        if ACTIVE.get() == 0 {
            live(true);
        }
        ACTIVE.set(ACTIVE.get() + 1);
        ExecGuard
    }
}

impl Drop for ExecGuard {
    fn drop(&mut self) {
        ACTIVE.set(ACTIVE.get() - 1);
        if ACTIVE.get() == 0 {
            live(false);
        }
    }
}

/// For every block version of every Cfg the executor runs (codes' own and
/// clones, current Plans only) that the optimizer made (blocks with an
/// origin): (qualname, clone depth, block, version, aten
/// ops, pinned calls, other calls, Autograd ops, body length).
#[pyfunction]
pub fn _cfg_region_stats() -> Vec<(String, u32, u32, u32, u32, u32, u32, u32, u32)> {
    let mut out = vec![];
    for c in registered() {
        let cfg = &c.plan().prep.cfg;
        for (b, blk) in cfg.blocks.iter().enumerate() {
            if blk.origin.is_none() {
                continue;
            }
            for (vi, ver) in blk.versions.iter().enumerate() {
                let (mut aten, mut pinned, mut calls, mut ag) = (0, 0, 0, 0);
                for v in &ver.body {
                    match &cfg[*v].op {
                        Op::Aten { .. } => aten += 1,
                        Op::Pinned(_) => pinned += 1,
                        Op::Call { .. } => calls += 1,
                        Op::Autograd(_) | Op::Region(_) => ag += 1,
                        _ => {}
                    }
                }
                out.push((c.qualname.clone(), c.depth, b as u32, vi as u32, aten, pinned, calls, ag, ver.body.len() as u32));
            }
        }
    }
    out
}

/// The CFG the executor runs for `code` (a code object, or a function via
/// `__code__`), with what was observed running it so far: the side effects at
/// each site (Python frames entered, torch ops run) and the type of each value.
/// `clone`: a path of instrs, each with a clone in the one before it (see
/// _cfg_clones), to show that clone's instead.
#[pyfunction]
#[pyo3(signature = (code, effects=true, types=true, clone=vec![]))]
pub fn cfg_observed(code: &Bound<'_, PyAny>, effects: bool, types: bool, clone: Vec<u32>) -> PyResult<String> {
    let c = unsafe { resolve(code, &clone)? };
    let plan = c.plan();
    let mut by_cfg = HashMap::new();
    let mut work = vec![&plan.prep];
    while let Some(p) = work.pop() {
        by_cfg.insert(Arc::as_ptr(&p.cfg), p);
        work.extend(p.origins.iter().flatten());
    }
    let observed = Observed { by_cfg, effects, types };
    Ok(plan.prep.cfg.render_with(false, false, &observed))
}

/// Optimize block `block` of the CFG the executor runs for `code` (a code
/// object, or a function via `__code__`); later calls run the result. See
/// cfg/optimize.rs.
#[pyfunction]
pub fn optimize_block(code: &Bound<'_, PyAny>, block: u32) -> PyResult<()> {
    let c = unsafe { lookup(code)? };
    optimize(code.py(), c, Some(BlockId(block)))
}

/// Specialize the region of the CFG the executor runs for `code` (of the
/// clone at `clone`) that starts at block `block`, along the directions its
/// branches were always seen to go (see cfg/specialize.rs).
#[pyfunction]
#[pyo3(signature = (code, block, clone=vec![]))]
pub fn _specialize(code: &Bound<'_, PyAny>, block: u32, clone: Vec<u32>) -> PyResult<()> {
    use crate::cfg::ir::Builtin;
    use types::{Kind, Konst, Ty};
    let c = unsafe { resolve(code, &clone)? };
    let plan = c.plan();
    let cfg = &plan.prep.cfg;
    let types = unsafe { &*plan.prep.types.get() };
    let mut dirs = HashMap::new();
    for b in cfg.block_ids() {
        if let (Terminator::Branch { cond, .. }, Some(ver)) = (&cfg[b].term, cfg[b].versions.first()) {
            let v = ver.outputs[*cond as usize];
            if let Some([Kind::Bool(Konst::Is(d))]) = types.get(v.idx()).and_then(|s| s.ty.as_ref()).and_then(Ty::kinds) {
                dirs.insert(b, *d);
            }
        }
    }
    // ToBools and Compares of values always seen an exact bool, int, float,
    // str or None.
    let exact_check = |v: ValueId| match types.get(v.idx()).and_then(|s| s.ty.as_ref()).and_then(Ty::kinds) {
        Some([Kind::Bool(_)]) => Some(Check::Exact(Builtin::Bool)),
        Some([Kind::Int(_)]) => Some(Check::Exact(Builtin::Int)),
        Some([Kind::Float]) => Some(Check::Exact(Builtin::Float)),
        Some([Kind::Str(_)]) => Some(Check::Exact(Builtin::Str)),
        Some([Kind::None]) => Some(Check::Value(Lit::None)),
        _ => None,
    };
    let mut exact = HashMap::new();
    for v in cfg.blocks.iter().filter_map(|b| b.versions.first()).flat_map(|ver| &ver.body) {
        // A compare of exact builtins is a bool.
        let compare_of_exact = |a: ValueId| matches!(cfg[a].op, Op::Compare(_)) && cfg[a].args.iter().all(|x| exact_check(*x).is_some());
        if matches!(cfg[*v].op, Op::ToBool) && compare_of_exact(cfg[*v].args[0]) {
            exact.insert(*v, vec![]);
        } else if matches!(cfg[*v].op, Op::ToBool | Op::Compare(_)) {
            if let Some(checks) = cfg[*v].args.iter().map(|a| exact_check(*a)).collect::<Option<Vec<_>>>() {
                exact.insert(*v, checks);
            }
        }
    }
    let mut new = (**cfg).clone();
    let renumbered = crate::cfg::specialize::specialize(&mut new, BlockId(block), &dirs, &exact).map_err(pyo3::exceptions::PyValueError::new_err)?;
    crate::cfg::verify::verify(&new).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("CFG verification failed: {e}")))?;
    swap_plan(c, plan, plan, new, plan.prep.record, vec![], HashMap::new(), |p| unsafe { &mut *p.effects.get() }.renumber_blocks(&renumbered))
}

thread_local! {
    // Set while optimizing, which runs Python code (see in_graph) that
    // mustn't be optimized itself meanwhile.
    static OPTIMIZING: Cell<bool> = const { Cell::new(false) };
}

// Optimizes `block` of c's Cfg, or every block it can (others are left as
// they are), and swaps in the result. Blocks are renumbered, so block-site
// effects move with their blocks.
// Runs outside the executor: the Python code it runs (see is_in_graph) would
// otherwise be recorded as effects of whatever site is running (the call
// whose callee just returned, for the optimization after a first run).
fn optimize(py: Python<'_>, c: &Compiled, block: Option<BlockId>) -> PyResult<()> {
    outside_executor(|| optimize_now(py, c, block, false))
}

/// What the executor holds for the code it ran, as (what, how many): code
/// objects; clones of them (one per call site that entered a Python
/// function, see Prepared::clones); those among both whose frames run on
/// CPython (see _cfg_bypass_cold); the plans of all of these; the instrs of
/// the plans' CFGs, and of the CFGs of their blocks' origins; the bytes of
/// the text every CFG carries (the reprs of the code's constants, its
/// names).
#[pyfunction]
pub fn _cfg_plan_stats() -> Vec<(&'static str, usize)> {
    fn origins(p: &Prepared) -> usize {
        p.origins.iter().flatten().map(|o| o.cfg.instrs.len() + origins(o)).sum()
    }
    fn text(cfg: &Cfg) -> usize {
        cfg.const_reprs.iter().chain(&cfg.names).chain(&cfg.slot_names).map(|s| s.len() + std::mem::size_of::<String>()).sum()
    }
    let (mut codes, mut clones, mut cold, mut plans, mut instrs, mut in_origins, mut bytes) = (0, 0, 0, 0, 0, 0, 0);
    for c in registered() {
        if c.depth == 0 {
            codes += 1;
        } else {
            clones += 1;
        }
        if c.cold.load(Ordering::Relaxed) {
            cold += 1;
        }
        let ps = c.plans.lock().unwrap();
        plans += ps.len();
        for p in ps.iter() {
            instrs += p.prep.cfg.instrs.len();
            in_origins += origins(&p.prep);
            bytes += text(&p.prep.cfg) + p.prep.origins.iter().flatten().map(|o| text(&o.cfg)).sum::<usize>();
        }
    }
    vec![("code objects", codes), ("clones", clones), ("cold", cold), ("plans", plans), ("instrs", instrs), ("instrs of origins", in_origins), ("bytes of text", bytes)]
}

/// Optimize every code the executor ran (and every clone) again, from what
/// they observed (see _reoptimize). Returns how many were.
#[pyfunction]
pub fn _reoptimize_all(py: Python<'_>) -> usize {
    let mut n = 0;
    for c in registered() {
        if !c.auto_optimized.load(Ordering::Relaxed) {
            continue;
        }
        match outside_executor(|| optimize_now(py, &c, None, true)) {
            Ok(()) => n += 1,
            Err(e) => {
                if std::env::var_os("TORCH_RUST_REGION_DEBUG").is_some() {
                    eprintln!("reoptimize {}: {e}", c.qualname);
                }
            }
        }
        // A round doesn't keep the round before.
        c.sweep();
    }
    n
}

// Whether _cfg_bypass_cold was called.
static BYPASSED: AtomicBool = AtomicBool::new(false);

/// Run the frames of every code (and clone) the executor ran whose Cfg has no
/// region (of a block or a span) or training version, nor any of its clones'
/// (recursively), on CPython from now on, and of code it hadn't run yet: the
/// executor alone runs Python slower. Returns how many there are.
#[pyfunction]
pub fn _cfg_bypass_cold() -> usize {
    BYPASSED.store(true, Ordering::Relaxed);
    fn hot(c: &Compiled, memo: &mut HashMap<usize, bool>) -> bool {
        let key = c as *const Compiled as usize;
        if let Some(h) = memo.get(&key) {
            return *h;
        }
        memo.insert(key, false);
        let prep = &c.plan().prep;
        let cfg = &prep.cfg;
        let spans = cfg.spans.iter().any(|s| !s.state.off.load(Ordering::Relaxed) && !s.state.regions.is_empty());
        let h = !cfg.regions.is_empty() || !cfg.trainings.is_empty() || spans || prep.clones.values().any(|cl| hot(&cl.compiled, memo));
        memo.insert(key, h);
        h
    }
    // Cold, and so is every frame its frames were seen entering: each a
    // clone's, deep too (not a callee found by code, which may be hot).
    fn deep(c: &Compiled, hot_memo: &mut HashMap<usize, bool>, memo: &mut HashMap<usize, bool>) -> bool {
        let key = c as *const Compiled as usize;
        if let Some(d) = memo.get(&key) {
            return *d;
        }
        memo.insert(key, true);
        let first = unsafe { &*(&*c.plans.lock().unwrap()[0] as *const Plan) };
        let clones = &c.plan().prep.clones;
        let effects = unsafe { &*first.prep.effects.get() };
        let d = c.flags & GENERATOR_FLAGS == 0
            && !hot(c, hot_memo)
            && effects.by_site().all(|(site, seqs)| {
                seqs.iter().flat_map(|(seq, _)| seq).all(|e| match e {
                    effects::Effect::Frame { how: effects::How::Direct, .. } => {
                        matches!(site, Site::Instr(v) if clones.get(&v).is_some_and(|cl| deep(&cl.compiled, hot_memo, memo)))
                    }
                    effects::Effect::Frame { .. } => false,
                    effects::Effect::TorchOp { .. } => true,
                })
            });
        memo.insert(key, d);
        d
    }
    let (mut memo, mut deep_memo) = (HashMap::new(), HashMap::new());
    let mut n = 0;
    for c in registered() {
        if c.auto_optimized.load(Ordering::Relaxed) && !hot(&c, &mut memo) {
            c.cold.store(true, Ordering::Relaxed);
            c.deep.store(deep(&c, &mut memo, &mut deep_memo), Ordering::Relaxed);
            n += 1;
        }
    }
    n
}

/// Optimize every block of the CFG the executor runs for `code` again, from
/// its unoptimized CFG, with everything observed so far (see reoptimize).
#[pyfunction]
#[pyo3(signature = (code, clone=vec![]))]
pub fn _reoptimize(code: &Bound<'_, PyAny>, clone: Vec<u32>) -> PyResult<()> {
    let c = unsafe { resolve(code, &clone)? };
    let r = outside_executor(|| optimize_now(code.py(), c, None, true));
    c.sweep();
    r
}

// With `again`, starts from the Cfg the code was converted to (the first
// Plan's) instead of the current one, so blocks optimized before can be
// optimized differently. Its value ids are the ones observations are kept
// by; values only later Cfgs had lose theirs, and so do blocks (block sites
// aren't kept in the first Plan's numbering), and the swap drops what the
// current Plan built since (training versions, origins' observations).
fn optimize_now(py: Python<'_>, c: &Compiled, block: Option<BlockId>, again: bool) -> PyResult<()> {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            OPTIMIZING.set(self.0);
        }
    }
    let _reset = Reset(OPTIMIZING.replace(true));
    let current = c.plan();
    // Read until the swap, across Python code that may let another thread
    // replace it.
    let _using = Using::new(current);
    // Plans are boxed, and the first lives as long as c.
    let plan: &Plan = if again { unsafe { &*(&*c.plans.lock().unwrap()[0] as *const Plan) } } else { current };
    if again {
        let n = plan.prep.cfg.instrs.len();
        unsafe {
            (*plan.prep.types.get()).truncate(n);
            (*plan.prep.callees.get()).retain(|v, _| v.idx() < n);
            (*plan.prep.effects.get()).retain(|s| matches!(s, Site::Instr(v) if v.idx() < n));
        }
    }
    let blocks: Vec<BlockId> = match block {
        Some(b) => vec![b],
        None => plan.prep.cfg.block_ids().collect(),
    };
    PURE_DEBUG_CODE.with_borrow_mut(|q| *q = format!("{}@{}", c.qualname, c.depth));
    PURE_DEBUG_LINES.with_borrow_mut(|l| l.clone_from(&c.lines));
    IMPURE_NOW.with_borrow_mut(|m| m.clear());
    let (mut movable, aten, pinnable, nulls) = observed_ops(py, &plan.prep, &blocks, Some(&current.prep.clones));
    let mut cfg = (*plan.prep.cfg).clone();
    let (mut objects, mut calls) = (HashMap::new(), HashMap::new());
    for (v, (f, call)) in pinnable {
        objects.insert(v, f);
        calls.insert(v, call);
    }
    // Calls whose clone's path can be copied in (only with ELIDE on).
    let mut inlines = HashMap::new();
    let mut el = Elisions { py, cfg: &mut cfg, pins: vec![], objects: HashMap::new(), next_id: 0 };
    if ELIDE.load(Ordering::Relaxed) {
        let consts = c.consts.bind(py).clone();
        let mut sites: Vec<ValueId> = current.prep.clones.keys().copied().filter(|v| v.idx() < plan.prep.cfg.instrs.len()).collect();
        sites.sort();
        for v in sites {
            if let Some(i) = el.inlinee(&plan.prep, &consts, v, &current.prep.clones[&v]) {
                if let Ok(d) = std::env::var("TORCH_RUST_ELIDE_DEBUG") {
                    let q = current.prep.clones[&v].code.bind(py).getattr("co_qualname").map_or("?".into(), |q| q.to_string());
                    if !d.is_empty() && d != "1" && !d.split(',').any(|x| x == q) {
                        continue;
                    }
                    eprintln!("elide: %{} ({q}) elided", v.0);
                }
                movable.insert(v);
                inlines.insert(v, i);
            }
        }
    }
    let (mut pins, inline_objects) = (std::mem::take(&mut el.pins), std::mem::take(&mut el.objects));
    // Generators suspend in their frames, which a region can't replay.
    let spannable = block.is_none() && c.flags & (0x20 | 0x80 | 0x200) == 0;
    let none = HashSet::new();
    let mut found = if SPANS.load(Ordering::Relaxed) && spannable { spans::find(py, &plan.prep, &movable, &current.prep.clones, &c.qualname, &[], &none) } else { vec![] };
    let hook = CFG_HOOK.lock().unwrap().as_ref().map(|h| h.clone_ref(py));
    let mut declared = HashSet::new();
    if let Some(hook) = hook.filter(|_| block.is_none()) {
        let view = cfg_view(py, c, &plan.prep, &movable, &found)?;
        let (more, forced, split) = cfg_edits(&hook.call1(py, (view,))?.into_bound(py))?;
        let changed = !more.is_empty() || !forced.is_empty() || !split.is_empty();
        declared = more.iter().filter_map(|v| plan.prep.cfg.instrs.get(v.idx())).filter(|i| is_call(&i.op)).map(|i| i.origin).collect();
        // What was said of them before the pass spoke no longer keeps
        // anything out of a region.
        let said: Vec<String> = IMPURE_NOW.with_borrow(|m| more.iter().filter_map(|v| m.get(&(&plan.prep as *const Prepared as usize, v.0)).cloned()).collect());
        if !said.is_empty() {
            let code = PURE_DEBUG_CODE.with_borrow(|q| q.clone());
            IMPURE.lock().unwrap().retain(|(c, _, w)| *c != code || !said.contains(w));
        }
        movable.extend(more);
        if changed && spannable {
            found = spans::find(py, &plan.prep, &movable, &current.prep.clones, &c.qualname, &forced, &split);
            if !SPANS.load(Ordering::Relaxed) {
                let entries: HashSet<BlockId> = forced.iter().map(|(e, _)| *e).collect();
                found.retain(|f| entries.contains(&f.entry));
            }
        }
    }
    if block.is_none() {
        *c.declared.lock().unwrap() = declared;
    }
    let obs = crate::cfg::optimize::Observed { movable, aten, calls, nulls };
    let (mut pinned, mut elided) = (vec![], vec![]);
    let renumbered = crate::cfg::optimize::optimize_blocks(&mut cfg, &blocks, &obs, &inlines, &mut pinned, &mut elided, block.is_some())
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    cfg.spans = spans::place(&cfg, found, &renumbered, &c.qualname);
    for (id, x, new) in elided {
        if let Some(o) = inline_objects.get(&(id, x)) {
            objects.insert(new, o.clone_ref(py));
        }
    }
    pins.extend(pinned.iter().map(|v| objects[v].clone_ref(py)));
    crate::cfg::verify::verify(&cfg).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("CFG verification failed: {e}")))?;
    let record = block.is_some() && plan.prep.record;
    let clones = new_clones(py, c, plan, current);
    swap_plan(c, current, plan, cfg, record, pins, clones, |p| unsafe { &mut *p.effects.get() }.renumber_blocks(&renumbered))
}

// Clones for the instrs of `plan`'s top-level blocks that entered a Python
// function directly on every visit (see Prepared::clones), other than those
// `current` has. Generators aren't cloned (a resumed generator frame finds its
// Compiled by code object).
fn new_clones(py: Python<'_>, c: &Compiled, plan: &Plan, current: &Plan) -> HashMap<ValueId, SiteClone> {
    let mut out = HashMap::new();
    if c.depth >= MAX_CLONE_DEPTH {
        return out;
    }
    let prep = &plan.prep;
    let (callees, effects) = unsafe { (&*prep.callees.get(), &*prep.effects.get()) };
    for v in prep.cfg.blocks.iter().flat_map(|b| &b.versions).flat_map(|ver| &ver.body) {
        if current.prep.clones.contains_key(v) {
            continue;
        }
        let Some(f) = callees.get(v).and_then(|c| c.get(py)) else { continue };
        let site = Site::Instr(*v);
        let seqs = effects.get(site);
        let entered = !effects.varies(site)
            && !seqs.is_empty()
            && seqs.iter().all(|(seq, _)| {
                seq.iter().filter(|e| matches!(e, effects::Effect::Frame { how: effects::How::Direct, .. })).count() == 1
            });
        if !entered || unsafe { pyo3::ffi::PyFunction_Check(f.as_ptr()) } == 0 {
            continue;
        }
        let Ok(code) = f.getattr("__code__") else { continue };
        let Ok(mut clone) = Compiled::new(&code) else { continue };
        if clone.flags & GENERATOR_FLAGS != 0 {
            continue;
        }
        clone.depth = c.depth + 1;
        clone.path = c.path.iter().cloned().chain([(c.qualname.clone(), c.line(prep.cfg[*v].origin))]).collect();
        let clone = Arc::new(clone);
        register(&clone);
        out.insert(*v, SiteClone { code: code.unbind(), compiled: clone });
    }
    out
}

// Builds the elisions of calls in `cfg` (see cfg/inline.rs): for each call
// with a clone whose observed path can be copied into the caller, its
// Inlinee, with the objects it pins appended to cfg.pins and to `pins`, and
// what the callee's pinned calls pin, by (inlinee id, callee instr), added to
// `objects`. The calls are made movable.
struct Elisions<'a> {
    py: Python<'a>,
    cfg: &'a mut Cfg,
    pins: Vec<Py<PyAny>>,
    objects: HashMap<(usize, ValueId), Py<PyAny>>,
    next_id: usize,
}

impl Elisions<'_> {
    // An object pinned before (by identity) keeps its pin, so reads of it
    // are the same read (see cfg/guard.rs).
    fn pin(&mut self, name: String, obj: Bound<'_, PyAny>) -> u32 {
        let base = self.cfg.pins.len() - self.pins.len();
        if let Some(k) = self.pins.iter().position(|p| p.is(&obj)) {
            return (base + k) as u32;
        }
        self.cfg.pins.push(crate::cfg::ir::Pin { name, path: None });
        self.pins.push(obj.unbind());
        self.cfg.pins.len() as u32 - 1
    }

    fn name(&mut self, n: &str) -> u32 {
        match self.cfg.names.iter().position(|x| x == n) {
            Some(i) => i as u32,
            None => {
                self.cfg.names.push(n.to_string());
                self.cfg.names.len() as u32 - 1
            }
        }
    }

    // `prep`: the caller's (its unoptimized Plan's, whose ids `v` is in);
    // `consts`: its code's co_consts.
    fn inlinee(&mut self, prep: &Prepared, consts: &Bound<'_, PyTuple>, v: ValueId, clone: &SiteClone) -> Option<crate::cfg::inline::Inlinee> {
        let r = self.try_inlinee(prep, consts, v, clone);
        if r.is_none() && std::env::var_os("TORCH_RUST_ELIDE_DEBUG").is_some() {
            let q = clone.code.bind(self.py).getattr("co_qualname").map_or("?".into(), |q| q.to_string());
            eprintln!("elide: %{} ({q}) refused: {}", v.0, ELIDE_WHY.take().unwrap_or_default());
        }
        r
    }

    fn try_inlinee(&mut self, prep: &Prepared, consts: &Bound<'_, PyTuple>, v: ValueId, clone: &SiteClone) -> Option<crate::cfg::inline::Inlinee> {
        use crate::cfg::inline::{Bind, Callable, Inlinee, Site as How};
        use types::{Kind, Konst, Ty};
        let py = self.py;
        let ins = prep.cfg[v].clone();
        let callee = unsafe { &*prep.callees.get() }.get(&v)?;
        let f = callee.get(py)?;
        let code = f.getattr("__code__").ok()?;
        if !code.is(clone.code.bind(py)) {
            return None;
        }
        let types = unsafe { &*prep.types.get() };
        let kinds = |x: ValueId| types.get(x.idx()).and_then(|s| s.ty.as_ref()).and_then(Ty::kinds);
        let is_class = |x: ValueId, tp: *const pyo3::ffi::PyTypeObject| matches!(kinds(x), Some([Kind::Object(c)]) if c.is(tp));
        // How the instr calls f, how many positional args (counting self) it
        // passes, and the keyword names.
        let (site, npos, kw): (How, usize, Vec<String>) = match (&ins.op, callee.via()?) {
            (Op::Call { kw_names, .. }, via) => {
                let kw: Vec<String> = match kw_names {
                    Some(k) => consts.get_item(*k as usize).ok()?.extract().ok()?,
                    None => vec![],
                };
                let method = ins.args[0] != crate::cfg::ir::NULL && !matches!(kinds(ins.args[0]), Some([Kind::Null]));
                let n = ins.args.len() - 2 + method as usize;
                let callable = match via {
                    effects::Via::Call => Callable::Instance,
                    effects::Via::Callable if method => Callable::Function,
                    effects::Via::Callable if is_class(ins.args[1], &raw const ffi::PyMethod_Type) => Callable::BoundMethod,
                    effects::Via::Callable => Callable::Function,
                    effects::Via::Getattr | effects::Via::Partial => return None,
                };
                let n = n + matches!(callable, Callable::Instance | Callable::BoundMethod) as usize;
                (How::Call { ex: false, method, callable }, n.checked_sub(kw.len())?, kw)
            }
            (Op::Bytecode { opcode: opcodes::CALL_FUNCTION_EX, .. }, via) => {
                let n = match kinds(ins.args[1]) {
                    Some([Kind::Tuple(items)]) => items.len(),
                    _ => return None,
                };
                if ins.args.len() == 3 && !matches!(kinds(ins.args[2]), Some([Kind::Dict(len)]) if *len == types::Interval::exact(0)) {
                    return None;
                }
                let callable = match via {
                    effects::Via::Call => Callable::Instance,
                    effects::Via::Callable if is_class(ins.args[0], &raw const ffi::PyMethod_Type) => Callable::BoundMethod,
                    effects::Via::Callable => Callable::Function,
                    effects::Via::Getattr | effects::Via::Partial => return None,
                };
                let n = n + matches!(callable, Callable::Instance | Callable::BoundMethod) as usize;
                (How::Call { ex: true, method: false, callable }, n, vec![])
            }
            (Op::LoadAttr(n) | Op::LoadMethod(n), effects::Via::Getattr) => {
                let attr = prep.cfg.names[*n as usize].clone();
                let s = pyo3::types::PyString::intern(py, &attr).into_any();
                let site = How::Getattr { name: self.name(&attr), str: self.pin(format!("{attr:?}"), s) };
                (site, 2, vec![])
            }
            _ => return None,
        };
        let attr = |n: &str| code.getattr(n).ok();
        let int = |n: &str| attr(n).and_then(|x| x.extract::<usize>().ok());
        let (argcount, posonly, kwonly) = (int("co_argcount")?, int("co_posonlyargcount")?, int("co_kwonlyargcount")?);
        let flags = int("co_flags")?;
        let (varargs, varkw) = (flags & 0x04 != 0, flags & 0x08 != 0);
        // Cells are fine (they become values); free variables aren't.
        if attr("co_freevars").and_then(|x| x.len().ok()) != Some(0) {
            return None;
        }
        let varnames: Vec<String> = attr("co_varnames")?.extract().ok()?;
        let named = argcount + kwonly;
        if npos > argcount && !varargs {
            return None;
        }
        let mut bind: Vec<Option<Bind>> = (0..named).map(|i| (i < npos).then_some(Bind::Arg(i))).collect();
        for (j, name) in kw.iter().enumerate() {
            let i = varnames[..named].iter().position(|n| n == name).filter(|i| *i >= posonly)?;
            if bind[i].is_some() {
                return None;
            }
            bind[i] = Some(Bind::Arg(npos + j));
        }
        let defaults = f.getattr("__defaults__").ok().filter(|d| !d.is_none());
        let nd = defaults.as_ref().map_or(0, |d| d.len().unwrap_or(0));
        let mut used_defaults = false;
        for (i, slot) in bind.iter_mut().enumerate() {
            if slot.is_none() {
                if !(i < argcount && i + nd >= argcount) {
                    return None;
                }
                let d = defaults.as_ref()?.get_item(i + nd - argcount).ok()?;
                let name = d.repr().map_or("?".into(), |r| r.to_string());
                *slot = Some(Bind::Pin(self.pin(name, d)));
                used_defaults = true;
            }
        }
        let mut bind: Vec<Bind> = bind.into_iter().map(|b| b.expect("bound")).collect();
        if varargs {
            bind.push(Bind::VarArgs(argcount.min(npos)));
        }
        if varkw {
            bind.push(Bind::VarKw);
        }
        let nslots = bind.len() as u32;

        // The callee's path, from what its clone observed.
        let base = unsafe { &*(&*clone.compiled.plans.lock().unwrap()[0] as *const Plan) };
        let bprep = &base.prep;
        let blocks: Vec<BlockId> = bprep.cfg.block_ids().collect();
        let (movable, aten, pinnable, nulls) = observed_ops(py, bprep, &blocks, None);
        let (btypes, beffects) = unsafe { (&*bprep.types.get(), &*bprep.effects.get()) };
        // Reads that ran no Python code and no torch op when observed (an
        // instance's attribute, a dict lookup): fine to copy in, since the
        // copy still does the read (nothing guards the class yet).
        let quiet = |x: ValueId| {
            let site = Site::Instr(x);
            !beffects.varies(site) && beffects.get(site).iter().all(|(seq, _)| seq.is_empty())
        };
        let cfg = bprep.cfg.clone();
        // What a load chain in the callee (a global, then module attributes)
        // gives now, read through f's globals; and the in-graph calls of such
        // a chain that ran nothing observable, which the caller may pin.
        let fglobals = f.getattr("__globals__").ok()?;
        let fbuiltins = f.getattr("__builtins__").ok()?;
        fn resolve<'py>(cfg: &Cfg, x: ValueId, g: &Bound<'py, PyAny>, bi: &Bound<'py, PyAny>) -> Option<Bound<'py, PyAny>> {
            let ins = &cfg[x];
            match ins.op {
                Op::LoadGlobal(n) => {
                    let name = &cfg.names[n as usize];
                    g.get_item(name).ok().or_else(|| bi.get_item(name).ok())
                }
                Op::LoadAttr(n) | Op::LoadMethod(n) => {
                    let m = resolve(cfg, ins.args[0], g, bi)?;
                    m.is_exact_instance_of::<pyo3::types::PyModule>().then_some(())?;
                    m.getattr(cfg.names[n as usize].as_str()).ok()
                }
                Op::Project(1) => resolve(cfg, ins.args[0], g, bi),
                _ => None,
            }
        }
        let mut quiet_calls: HashMap<ValueId, crate::cfg::optimize::PinnedCall> = HashMap::new();
        let mut quiet_objects: Vec<(ValueId, Py<PyAny>)> = vec![];
        let cprep = &clone.compiled.plan().prep;
        let deeper = clone.compiled.depth < MAX_CLONE_DEPTH;
        let mut dirs = HashMap::new();
        let mut nested = HashMap::new();
        let mut defined: HashSet<u32> = (0..nslots).collect();
        let mut seen = HashSet::new();
        let mut b = crate::cfg::ir::ENTRY;
        let cconsts = clone.compiled.consts.bind(py).clone();
        loop {
            if !seen.insert(b) {
                return None;
            }
            let blk = &cfg[b];
            let [ver] = &blk.versions[..] else { return None };
            for x in &ver.body {
                let op = &cfg[*x].op;
                let mut nest = |me: &mut Self| match cprep.clones.get(x).filter(|_| deeper) {
                    Some(sub) => match me.inlinee(bprep, &cconsts, *x, sub) {
                        Some(i) => {
                            nested.insert(*x, i);
                            true
                        }
                        None => false,
                    },
                    None => false,
                };
                let ok = match op {
                    Op::Resume | Op::Project(_) | Op::MakeCell { .. } => true,
                    Op::LoadLocal { slot } | Op::LoadLocalChecked { slot } | Op::LoadDeref { slot } => defined.contains(slot),
                    Op::StoreLocal { slot } | Op::StoreDeref { slot } => {
                        defined.insert(*slot);
                        true
                    }
                    Op::Bytecode { opcode: opcodes::BUILD_MAP, arg: 0 } | Op::Bytecode { opcode: opcodes::DICT_MERGE, arg: 1 } => true,
                    Op::LoadAttr(_) | Op::LoadMethod(_) => movable.contains(x) || quiet(*x) || nest(self),
                    Op::LoadGlobal(_) | Op::Const(_) | Op::Binary(_) | Op::Compare(_) | Op::BuildTuple(_) | Op::StoreAttr(_) => movable.contains(x),
                    Op::ToBool | Op::IsNone => movable.contains(x) || quiet(*x),
                    Op::Bytecode { opcode, .. } => {
                        matches!(
                            *opcode,
                            opcodes::UNARY_NEGATIVE | opcodes::UNARY_NOT | opcodes::UNARY_INVERT | opcodes::BINARY_SUBSCR | opcodes::IS_OP | opcodes::CONTAINS_OP | opcodes::BUILD_LIST
                        ) && (movable.contains(x) || quiet(*x))
                            || *opcode == opcodes::CALL_FUNCTION_EX && (movable.contains(x) || nest(self))
                    }
                    Op::Call { .. } => {
                        let a = &cfg[*x].args;
                        let target = resolve(&cfg, a[1], &fglobals, &fbuiltins).filter(|_| a[0] == crate::cfg::ir::NULL || matches!(cfg[a[0]].op, Op::Project(0)));
                        match target {
                            Some(t) if quiet(*x) && is_in_graph(&t) => {
                                let name = t.getattr("__qualname__").map_or("?".into(), |q| q.to_string());
                                let has_torch_function = is_has_torch_function(&t);
                                quiet_calls.insert(*x, crate::cfg::optimize::PinnedCall { name, args: None, has_torch_function });
                                quiet_objects.push((*x, t.unbind()));
                                true
                            }
                            _ => movable.contains(x) || nest(self),
                        }
                    }
                    _ => false,
                };
                if !ok {
                    if ELIDE_WHY.with_borrow(|w| w.is_none()) {
                        why(format!("{:?} %{} in bb{}", op, x.0, b.0));
                    }
                    return None;
                }
            }
            let next = match &blk.term {
                Terminator::Jump(e) => e,
                Terminator::Branch { cond, then, else_ } => {
                    let c = ver.outputs[*cond as usize];
                    let dir = match btypes.get(c.idx()).and_then(|s| s.ty.as_ref()).and_then(Ty::kinds) {
                        Some([Kind::Bool(Konst::Is(d))]) => *d,
                        _ => {
                            why(format!("bb{}'s branch went both ways", b.0));
                            return None;
                        }
                    };
                    dirs.insert(b, dir);
                    if dir { then } else { else_ }
                }
                _ => return None,
            };
            match next {
                Exit::Return(_) => break,
                Exit::Block(e) => b = e.target,
                Exit::Leave(_) => return None,
            }
        }

        let id = self.next_id;
        self.next_id += 1;
        let mut calls = HashMap::new();
        for (x, (obj, call)) in pinnable {
            self.objects.insert((id, x), obj);
            calls.insert(x, call);
        }
        for (x, obj) in quiet_objects {
            self.objects.insert((id, x), obj);
        }
        calls.extend(quiet_calls);
        let function = self.pin(callee.describe(py), f.clone());
        let defaults = if used_defaults {
            let d = defaults.expect("defaults were bound");
            let pin = self.pin("__defaults__".into(), d);
            Some((pin, self.name("__defaults__")))
        } else {
            None
        };
        let globals = self.pin("globals".into(), f.getattr("__globals__").ok()?);
        let builtins = self.pin("builtins".into(), f.getattr("__builtins__").ok()?);
        let mut cs = vec![];
        for k in cconsts.iter() {
            let name = k.repr().map_or("?".into(), |r| r.to_string());
            cs.push(self.pin(name, k));
        }
        let names = [self.name("__func__"), self.name("__self__"), self.name("__call__")];
        let obs = crate::cfg::optimize::Observed { movable: Default::default(), aten, calls, nulls };
        Some(Inlinee { id, obs, cfg, dirs, site, bind, function, defaults, globals, builtins, consts: cs, nested, names })
    }
}

// Makes `cfg`, a changed copy of `base`'s, the Cfg new frames of c run in
// place of `current`'s (usually base). `before` runs first, unless another
// thread replaced `current` meanwhile. `pins`: what the Cfg's pins beyond
// base's resolve to.
#[allow(clippy::too_many_arguments)]
fn swap_plan(
    c: &Compiled,
    current: &Plan,
    base: &Plan,
    cfg: Cfg,
    record: bool,
    pins: Vec<Py<PyAny>>,
    clones: HashMap<ValueId, SiteClone>,
    before: impl FnOnce(&Prepared),
) -> PyResult<()> {
    let mut plans = c.plans.lock().unwrap();
    if !std::ptr::eq(c.plan(), current) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err("the CFG was optimized concurrently"));
    }
    before(&base.prep);
    let mut new = Box::new(Plan::new(cfg, Some(base), record, pins, &c.consts, &c.names)?);
    // The Plans replaced before this one that no frame runs go (see
    // Compiled::plan). The one being replaced stays until the next sweep:
    // the caller still reads it, and so may `base`.
    let mut first = true;
    plans.retain(|p| {
        let keep = first || std::ptr::eq(&**p, current) || std::ptr::eq(&**p, base) || p.users.load(Ordering::Relaxed) > 0 || p.prep.kept.load(Ordering::Relaxed);
        first = false;
        keep
    });
    Python::attach(|py| {
        for (v, cl) in &current.prep.clones {
            new.prep.clones.insert(*v, SiteClone { code: cl.code.clone_ref(py), compiled: cl.compiled.clone() });
        }
    });
    new.prep.clones.extend(clones);
    c.plan.store(&mut *new, Ordering::Release);
    plans.push(new);
    Ok(())
}

// Gives block b of code's Cfg a version running `t` (see autograd.rs), if
// the Cfg is still the one whose Prepared is at `prep`.
pub(crate) fn add_training(c: &Compiled, prep: usize, b: BlockId, checks: Vec<(Option<u32>, Check)>, t: Training) -> PyResult<()> {
    let plan = c.plan();
    if &plan.prep as *const Prepared as usize != prep {
        return Err(pyo3::exceptions::PyRuntimeError::new_err("the CFG changed since the block ran"));
    }
    let mut cfg = (*plan.prep.cfg).clone();
    crate::cfg::partition::add_training_version(&mut cfg, b, checks, t)
        .map_err(|e| pyo3::exceptions::PyNotImplementedError::new_err(format!("no training version for bb{}: {e}", b.0)))?;
    crate::cfg::verify::verify(&cfg).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("CFG verification failed: {e}")))?;
    swap_plan(c, plan, plan, cfg, plan.prep.record, vec![], HashMap::new(), |_| {})?;
    // The blocks keep their ids: those that gave up stay given up.
    let new = &c.plan().prep;
    for (i, flag) in plan.prep.autograd.iter().enumerate() {
        if !flag.load(Ordering::Relaxed) && new.autograd[i].swap(false, Ordering::Relaxed) {
            unsafe { *new.autograd_why[i].get() = (*plan.prep.autograd_why[i].get()).clone() };
        }
    }
    Ok(())
}

// Stops block b of code's Cfg from bootstrapping (it can't get a training
// version), if the Cfg is still the one whose Prepared is at `prep`.
pub(crate) fn no_training(c: &Compiled, prep: usize, b: BlockId, why: String) {
    let plan = c.plan();
    if &plan.prep as *const Prepared as usize == prep {
        plan.prep.autograd[b.idx()].store(false, Ordering::Relaxed);
        unsafe { *plan.prep.autograd_why[b.idx()].get() = Some(why) };
    }
}

// The Cfg whose regions a frame running `prep`'s sees or adds to: c's
// current one, if it is `prep`'s or derives from it by adding regions (which
// leaves the blocks and their other versions as they were).
pub(crate) fn region_cfg(c: &Compiled, prep: usize) -> Option<&Cfg> {
    let plan = c.plan();
    let here = &plan.prep as *const Prepared as usize;
    let lineage = unsafe { &*plan.prep.region_lineage.get() };
    (here == prep || lineage.contains(&prep)).then(|| &*plan.prep.cfg)
}

// The regions of block b of `cfg`, if it has any: what its region version
// (the first) runs.
pub(crate) fn place_of(cfg: &Cfg, b: BlockId) -> Option<&Arc<crate::cfg::ir::Place>> {
    let head = cfg[b].versions.first()?.body.first()?;
    match cfg[*head].op {
        Op::Region(r) => cfg.regions.get(r as usize),
        _ => None,
    }
}

// Gives block b of code's Cfg a region (see region.rs) made from a trace of
// its last version, if the Cfg is the one whose Prepared is at `prep` or
// derives from it by adding regions. The block's first region comes with a
// region version in front of its others, with the guard of the version the
// trace was made from; later ones are added to that version's place, which
// changes no Cfg. Ok(false): the Cfg is another (trace it again).
pub(crate) fn add_region(c: &Compiled, prep: usize, b: BlockId, traced: region::Traced) -> PyResult<bool> {
    let plan = c.plan();
    let here = &plan.prep as *const Prepared as usize;
    let lineage = unsafe { &*plan.prep.region_lineage.get() };
    if here != prep && !lineage.contains(&prep) {
        return Ok(false);
    }
    let n = traced.variant.spec.outputs.len();
    if let Some(place) = place_of(&plan.prep.cfg, b) {
        if place.outputs != n {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!("a region of {} has {n} outputs, not {}", place.name, place.outputs)));
        }
        place.variants.push(traced.variant);
        return Ok(true);
    }
    let mut cfg = (*plan.prep.cfg).clone();
    let vi = cfg[b].versions.len() - 1;
    let handler = region_handler(&cfg, b, vi).map_err(pyo3::exceptions::PyNotImplementedError::new_err)?;
    let blk = &cfg[b];
    let from = &blk.versions[vi];
    let guard = from.guard.clone();
    let inputs = from.inputs.clone();
    let args: Vec<ValueId> = blk.params.iter().chain(&inputs).copied().collect();
    let origin = from.body.first().map_or(0, |v| cfg[*v].origin);
    let place = crate::cfg::ir::Place { name: traced.variant.spec.name.clone(), outputs: n, variants: Default::default() };
    place.variants.push(traced.variant);
    cfg.regions.push(Arc::new(place));
    let v = cfg.add(Op::Region(cfg.regions.len() as u32 - 1), args, origin);
    let outputs: Vec<ValueId> = (0..n).map(|k| cfg.add(Op::Project(k as u32), vec![v], origin)).collect();
    let mut body = vec![v];
    body.extend(outputs.iter().copied());
    let unwind = match handler {
        crate::cfg::ir::Handler::Propagate => vec![],
        handler => vec![crate::cfg::ir::UnwindSegment { from: 0, handler }],
    };
    cfg[b].versions.insert(0, crate::cfg::ir::Version { guard, body, outputs, unwind, inputs });
    crate::cfg::verify::verify(&cfg).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("CFG verification failed: {e}")))?;
    swap_plan(c, plan, plan, cfg, plan.prep.record, vec![], HashMap::new(), |_| {})?;
    let new = &c.plan().prep;
    for (i, flag) in plan.prep.region_off.iter().enumerate() {
        new.region_off[i].store(flag.load(Ordering::Relaxed), Ordering::Relaxed);
        unsafe { *new.region_types[i].get() = (*plan.prep.region_types[i].get()).clone() };
    }
    unsafe { *new.region_lineage.get() = lineage.iter().copied().chain([here]).collect() };
    Ok(true)
}

// Where a raise in a region version of block b (made from version vi) goes:
// the one handler vi's ops raise into, taking only what the block was given.
// A landing block that only stores pending locals and re-raises out of the
// frame counts as propagating (a region's raise leaves those locals stale).
fn region_handler(cfg: &Cfg, b: BlockId, vi: usize) -> Result<crate::cfg::ir::Handler, String> {
    use crate::cfg::ir::{Handler, TermUnwind};
    let blk = &cfg[b];
    let from = &blk.versions[vi];
    let landing = |k: usize| {
        let h = &cfg[blk.handlers[k]];
        h.versions.len() == 1
            && h.versions[0].body.iter().all(|v| matches!(cfg[*v].op, Op::StoreLocal { .. }))
            && matches!(&h.term, Terminator::Reraise { unwind: TermUnwind::Propagate, .. })
    };
    let given = |args: &[ValueId]| args.iter().all(|a| blk.params.contains(a) || from.inputs.contains(a) || a.is_reserved());
    let mut out: Option<Handler> = None;
    for seg in &from.unwind {
        let h = match &seg.handler {
            Handler::Propagate => Handler::Propagate,
            Handler::Block(k, _) if landing(*k) => Handler::Propagate,
            Handler::Block(_, args) | Handler::Outer(_, args) if given(args) => seg.handler.clone(),
            _ => return Err("it raises into a handler taking values it made".into()),
        };
        match &out {
            Some(o) if *o != h => return Err(format!("its ops raise into several handlers: {o:?} and {h:?}")),
            _ => out = Some(h),
        }
    }
    Ok(out.unwrap_or(Handler::Propagate))
}

// A check a region's trace asks for (torch._rust_inductor._plan), given as a
// tuple (kind, position among the place's args (or a value for global
// checks), ...): the position with the check, or with the object the arg is
// to be.
pub(crate) enum Parsed {
    Check(Option<usize>, Check),
    Is(usize, Py<PyAny>),
}

pub(crate) fn parse_check_tuple(t: &Bound<'_, PyTuple>) -> PyResult<Parsed> {
    let bad = || pyo3::exceptions::PyValueError::new_err(format!("bad check {t}"));
    let kind: String = t.get_item(0)?.extract()?;
    let arg = || -> PyResult<Option<usize>> { Ok(Some(t.get_item(1)?.extract()?)) };
    let (at, check) = match kind.as_str() {
        "grad_enabled" => (None, Check::GradEnabled(t.get_item(1)?.extract()?)),
        "no_torch_function_mode" => (None, Check::NoTorchFunctionMode),
        "no_forward_ad" => (None, Check::NoForwardAD),
        "no_saved_tensor_hooks" => (None, Check::NoSavedTensorHooks),
        "requires_grad" => (arg()?, Check::RequiresGrad(t.get_item(2)?.extract()?)),
        "plain_tensor" => (arg()?, Check::PlainTensor),
        "null" => (arg()?, Check::Null),
        "tensor" => {
            let x = t.get_item(2)?;
            let tensor: &crate::bindings::ffi::Tensor = x.extract()?;
            let mut m = crate::bindings::ffi::TensorMeta::default();
            crate::bindings::ffi::tensor_meta(tensor, &mut m);
            let spec = TensorSpec { dtype: m.dtype, device_type: m.device_type, device_index: m.device_index, sizes: tensor.sizes().to_vec(), strides: tensor.strides().to_vec() };
            (arg()?, Check::TensorMeta(Box::new(spec)))
        }
        "value" => (arg()?, Check::Value(lit(&t.get_item(2)?).ok_or_else(bad)?)),
        "ints" => (arg()?, Check::Ints(t.get_item(2)?.extract()?, t.get_item(3)?.extract()?)),
        "is" => return Ok(Parsed::Is(t.get_item(1)?.extract()?, t.get_item(2)?.unbind())),
        _ => return Err(bad()),
    };
    Ok(Parsed::Check(at, check))
}

/// For coverage reports: what became of each pure block of the CFG the
/// executor runs for `code` with autograd on, as (block, guard misses,
/// status). The status is "trained" (it has a training version), "eligible"
/// (it bootstraps when next run with grad needed), "gave up: why" (its
/// backward couldn't be compiled) or "ineligible: why".
#[pyfunction]
#[pyo3(signature = (code, clone=vec![]))]
pub fn _cfg_autograd_blocks(code: &Bound<'_, PyAny>, clone: Vec<u32>) -> PyResult<Vec<(u32, u32, String)>> {
    let c = unsafe { resolve(code, &clone)? };
    let prep = &c.plan().prep;
    let cfg = &prep.cfg;
    let mut out = vec![];
    for (i, blk) in cfg.blocks.iter().enumerate() {
        if blk.origin.is_none() {
            continue;
        }
        let b = BlockId(i as u32);
        let trained = blk.versions.iter().any(|v| v.body.iter().any(|x| matches!(cfg[*x].op, crate::cfg::ir::Op::Autograd(_))));
        let status = if trained {
            "trained".to_string()
        } else if let Some(why) = autograd::ineligible(cfg, b) {
            format!("ineligible: {why}")
        } else if prep.autograd[i].load(Ordering::Relaxed) {
            "eligible".to_string()
        } else {
            format!("gave up: {}", unsafe { &*prep.autograd_why[i].get() }.as_deref().unwrap_or("?"))
        };
        out.push((i as u32, prep.misses[i].load(Ordering::Relaxed), status));
    }
    Ok(out)
}

/// For testing guards: add checks to the first version of block `block` (which
/// must have an origin to fall back to) of the CFG the executor runs for
/// `code`. Each check is a tuple: ("grad_enabled", bool),
/// ("requires_grad", param, bool), ("tensor", param, example tensor),
/// ("value", param, None | bool | int | float), ("no_forward_ad",),
/// ("no_saved_tensor_hooks",), ("expect", param, bool) (the param is True
/// or False) and ("expect_is", param, object).
#[pyfunction]
pub fn _add_block_checks(code: &Bound<'_, PyAny>, block: u32, checks: Vec<Bound<'_, PyTuple>>) -> PyResult<()> {
    let c = unsafe { lookup(code)? };
    let plan = c.plan();
    let mut cfg = (*plan.prep.cfg).clone();
    let mut pins: Vec<Py<PyAny>> = vec![];
    let b = BlockId(block);
    if b.idx() >= cfg.blocks.len() || cfg[b].origin.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!("bb{block} has no origin")));
    }
    let bad = |c: &Bound<'_, PyTuple>| pyo3::exceptions::PyValueError::new_err(format!("bad check {c}"));
    for t in &checks {
        let kind: String = t.get_item(0)?.extract()?;
        let params = cfg[b].params.clone();
        let param = || -> PyResult<Option<ValueId>> {
            let i: usize = t.get_item(1)?.extract()?;
            params.get(i).copied().map(Some).ok_or_else(|| bad(t))
        };
        let check = match kind.as_str() {
            "grad_enabled" => (None, Check::GradEnabled(t.get_item(1)?.extract()?)),
            "requires_grad" => (param()?, Check::RequiresGrad(t.get_item(2)?.extract()?)),
            "tensor" => {
                let x = t.get_item(2)?;
                let tensor: &crate::bindings::ffi::Tensor = x.extract()?;
                let mut m = crate::bindings::ffi::TensorMeta::default();
                crate::bindings::ffi::tensor_meta(tensor, &mut m);
                let spec = TensorSpec {
                    dtype: m.dtype,
                    device_type: m.device_type,
                    device_index: m.device_index,
                    sizes: tensor.sizes().to_vec(),
                    strides: tensor.strides().to_vec(),
                };
                (param()?, Check::TensorMeta(Box::new(spec)))
            }
            "value" => (param()?, Check::Value(lit(&t.get_item(2)?).ok_or_else(|| bad(t))?)),
            "no_forward_ad" => (None, Check::NoForwardAD),
            "no_saved_tensor_hooks" => (None, Check::NoSavedTensorHooks),
            "expect" => (param()?, Check::Value(Lit::Bool(t.get_item(2)?.extract()?))),
            "expect_is" => {
                let obj = t.get_item(2)?;
                cfg.pins.push(crate::cfg::ir::Pin { name: obj.repr()?.to_string(), path: None });
                pins.push(obj.unbind());
                (param()?, Check::Is(cfg.pins.len() as u32 - 1))
            }
            _ => return Err(bad(t)),
        };
        cfg[b].versions[0].guard.checks.push(check);
    }
    crate::cfg::verify::verify(&cfg).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("CFG verification failed: {e}")))?;
    swap_plan(c, plan, plan, cfg, plan.prep.record, pins, HashMap::new(), |_| {})
}

// A scalar as a Lit (bool before int: bool is an int subclass).
pub(crate) fn lit(o: &Bound<'_, PyAny>) -> Option<Lit> {
    if o.is_none() {
        Some(Lit::None)
    } else if o.is_exact_instance_of::<pyo3::types::PyBool>() {
        Some(Lit::Bool(o.is_truthy().ok()?))
    } else if o.is_exact_instance_of::<pyo3::types::PyInt>() {
        Some(Lit::Int(o.extract().ok()?))
    } else if o.is_exact_instance_of::<pyo3::types::PyFloat>() {
        Some(Lit::Float(o.extract::<f64>().ok()?.to_bits()))
    } else {
        None
    }
}

/// How many times each block of the CFG the executor runs for `code` found
/// no version whose guard held (and ran its origin), since the CFG last
/// changed.
#[pyfunction]
#[pyo3(signature = (code, clone=vec![]))]
pub fn _cfg_misses(code: &Bound<'_, PyAny>, clone: Vec<u32>) -> PyResult<Vec<u32>> {
    let c = unsafe { resolve(code, &clone)? };
    Ok(c.plan().prep.misses.iter().map(|m| m.load(Ordering::Relaxed)).collect())
}

// The instrs of `blocks` that, as observed, run no code that could look at
// the frame or care when a local's old value is released, so they may sit in
// a pure block (see cfg/optimize.rs): roughly what Dynamo would put in a
// graph. Arithmetic, attribute and subscript ops count when their operands
// were only ever builtin scalars, strings, tensors, tuples or lists (or, for
// attributes, modules) and no Python frame was entered there; calls when they
// always called the same function, one Dynamo puts in its graphs. Also returns
// the operators that may be replaced by the aten op they ran: every visit ran
// that one op, the left operand was always a plain tensor (a reflected
// operator's aten args don't follow the operand order) and each operand was
// always a plain tensor or one builtin scalar type. And the movable calls
// whose callee may be pinned, with it: every visit ran one torch op, the same
// one (so a pinned call is like an aten op to autograd). And the values only
// ever seen NULL.
#[allow(clippy::type_complexity)]
fn observed_ops(
    py: Python<'_>,
    prep: &Prepared,
    blocks: &[BlockId],
    clones: Option<&HashMap<ValueId, SiteClone>>,
) -> (
    HashSet<ValueId>,
    HashMap<ValueId, crate::cfg::optimize::AtenOp>,
    HashMap<ValueId, (Py<PyAny>, crate::cfg::optimize::PinnedCall)>,
    HashSet<ValueId>,
) {
    use crate::cfg::ir::Builtin;
    use crate::cfg::optimize::{Arg, AtenOp, PinnedCall};
    use types::{Kind, Ty};
    let mut aten = HashMap::new();
    let cfg = &prep.cfg;
    // Calls to judge with is_in_graph, which runs Python code (observed by the
    // executor, so not while holding the tables' locks).
    // The last field: whether the call entered the callee's frame (and no
    // other), so the callee must be a leaf (see leaf_ops).
    let mut calls: Vec<(ValueId, Py<PyAny>, Option<Option<Vec<Option<Arg>>>>, bool)> = vec![];
    let mut out = HashSet::new();
    let mut nulls = HashSet::new();
    {
        let types = unsafe { &*prep.types.get() };
        let effects = unsafe { &*prep.effects.get() };
        let callees = unsafe { &*prep.callees.get() };
        let plain = |v: &ValueId, module_ok: bool| match types[v.idx()].ty.as_ref().and_then(Ty::kinds) {
            Some(ks) => ks.iter().all(|k| match k {
                Kind::None
                | Kind::Bool(_)
                | Kind::Int(_)
                | Kind::Float
                | Kind::Str(_)
                | Kind::Tensor(_)
                | Kind::Tuple(_)
                | Kind::VarTuple(_)
                | Kind::List(_) => true,
                Kind::Object(c) => module_ok && c.is(&raw const pyo3::ffi::PyModule_Type),
                _ => false,
            }),
            _ => false,
        };
        let in_range = || blocks.iter().filter(|b| b.idx() < cfg.blocks.len());
        for v in in_range().flat_map(|b| cfg[*b].params.iter().chain(cfg[*b].versions.iter().flat_map(|ver| &ver.body))) {
            if matches!(types[v.idx()].ty.as_ref().and_then(Ty::kinds), Some([Kind::Null])) {
                nulls.insert(*v);
            }
        }
        let bodies = in_range().flat_map(|b| cfg[*b].versions.first());
        let mut memo = HashMap::new();
        for v in bodies.flat_map(|ver| &ver.body) {
            let ins = &cfg[*v];
            let arg = |a: &ValueId| match types[a.idx()].ty.as_ref().and_then(Ty::kinds) {
                Some(ks) => match ks {
                    [Kind::Tensor(t)] if plain_tensor_types().iter().any(|p| t.class().is(*p as *const _)) => {
                        Some(Arg::PlainTensor)
                    }
                    [Kind::Int(_)] => Some(Arg::Scalar(Builtin::Int)),
                    [Kind::Float] => Some(Arg::Scalar(Builtin::Float)),
                    [Kind::Bool(_)] => Some(Arg::Scalar(Builtin::Bool)),
                    _ => None,
                },
                _ => None,
            };
            let no_frames = || {
                let site = Site::Instr(*v);
                !effects.varies(site) && effects.get(site).iter().all(|(seq, _)| !seq.iter().any(|e| matches!(e, effects::Effect::Frame { .. })))
            };
            // With pure calls on: a site whose every visit ran only torch ops
            // and frames of pure callees (see pure_site) is movable whatever
            // it is, as long as it writes nothing itself.
            let pure = clones.filter(|_| PURE_CALLS.load(Ordering::Relaxed)).is_some_and(|clones| {
                ins.op.can_raise() && pure_site(py, prep, *v, clones, &mut memo)
            });
            let ok = pure || match &ins.op {
                Op::LoadLocal { .. }
                | Op::LoadLocalChecked { .. }
                | Op::StoreLocal { .. }
                | Op::DeleteLocal { .. }
                | Op::Resume
                | Op::EvalBreaker
                | Op::LoadGlobal(_) => true,
                _ if callees.get(v).is_some_and(|c| c.via() == Some(effects::Via::Callable)) => {
                    let site = Site::Instr(*v);
                    let seqs = effects.get(site);
                    // Every visit entered the callee's frame and nothing else
                    // (what ran inside it is recorded in the callee's own Cfg).
                    let direct = !effects.varies(site)
                        && !seqs.is_empty()
                        && seqs.iter().all(|(seq, _)| matches!(seq.as_slice(), [effects::Effect::Frame { how: effects::How::Direct, .. }]));
                    let tensor_out = arg(v) == Some(Arg::PlainTensor);
                    let args = || tensor_out.then(|| ins.args.iter().skip(2).map(arg).collect());
                    if let (Some(f), true) = (callees[v].get(py), direct) {
                        calls.push((*v, f.unbind(), Some(args()), true));
                    } else if let (Some(f), true) = (callees[v].get(py), no_frames()) {
                        let seqs = effects.get(Site::Instr(*v));
                        let op = |seq: &[effects::Effect]| match seq {
                            [effects::Effect::TorchOp { op, .. }] => Some(op.clone()),
                            _ => None,
                        };
                        let first = seqs.first().and_then(|(seq, _)| op(seq));
                        let one_op = first.is_some() && seqs.iter().all(|(seq, _)| op(seq) == first);
                        // Or ran nothing observable (a predicate): pinnable
                        // too, though its result isn't a tensor.
                        let quiet = !seqs.is_empty() && seqs.iter().all(|(seq, _)| seq.is_empty());
                        calls.push((*v, f.unbind(), (one_op || quiet).then(|| if one_op { args() } else { None }), false));
                    }
                    false
                }
                op if !op.can_raise() => true,
                Op::LoadAttr(_) | Op::LoadMethod(_) => no_frames() && plain(&ins.args[0], true),
                // On plain objects, running no Python code.
                Op::StoreAttr(_) => {
                    let objects = matches!(types[ins.args[1].idx()].ty.as_ref().and_then(Ty::kinds), Some(ks) if ks.iter().all(|k| matches!(k, Kind::Object(_))));
                    no_frames() && objects
                }
                Op::Binary(_) | Op::Compare(_) | Op::ToBool | Op::GetIter | Op::Bytecode { .. } => {
                    no_frames() && ins.args.iter().all(|a| plain(a, false))
                }
                _ => false,
            };
            if ok {
                out.insert(*v);
            }
            let replaceable = matches!(
                ins.op,
                Op::Binary(_)
                    | Op::Compare(_)
                    | Op::Bytecode {
                        opcode: opcodes::UNARY_NEGATIVE | opcodes::UNARY_INVERT,
                        ..
                    }
            );
            if let (true, false, [(seq, _)]) = (replaceable, effects.varies(Site::Instr(*v)), effects.get(Site::Instr(*v))) {
                if let [effects::Effect::TorchOp { op, .. }] = seq.as_slice() {
                    let args: Option<Vec<Arg>> = ins.args.iter().map(arg).collect();
                    if let Some(args) = args.filter(|a| a.first() == Some(&Arg::PlainTensor)) {
                        aten.insert(*v, AtenOp { name: op.clone(), args });
                    }
                }
            }
        }
    }
    let mut in_graph: HashMap<*mut pyo3::ffi::PyObject, bool> = HashMap::new();
    let mut pinnable = HashMap::new();
    for (v, f, pin, entered) in calls {
        if entered && !is_leaf(f.bind(py)) {
            continue;
        }
        if *in_graph.entry(f.as_ptr()).or_insert_with(|| is_in_graph(f.bind(py))) {
            out.insert(v);
            if let Some(args) = pin {
                let attr = |n: &str| f.bind(py).getattr(n).and_then(|a| a.extract::<String>()).ok();
                let name = match (attr("__module__"), attr("__qualname__")) {
                    (Some(m), Some(q)) => format!("{m}.{q}"),
                    (_, q) => q.unwrap_or_else(|| "?".into()),
                };
                let has_torch_function = is_has_torch_function(f.bind(py));
                pinnable.insert(v, (f, PinnedCall { name, args, has_torch_function }));
            }
        }
    }
    (out, aten, pinnable, nulls)
}

// Whether optimizing treats calls (and attribute reads, subscripts, ...)
// whose callees were only ever seen running torch ops (see pure_compiled) as
// movable, so pure blocks span them; their runs are what regions trace.
static PURE_CALLS: AtomicBool = AtomicBool::new(false);
static SPANS: AtomicBool = AtomicBool::new(false);

// torch.vm's cfg passes: called (see cfg_view) on each Cfg about to be
// optimized, returning edits.
static CFG_HOOK: Mutex<Option<Py<PyAny>>> = Mutex::new(None);

/// Call `hook(view)` on each function's CFG before it's optimized (see
/// torch.vm.Cfg); None stops.
#[pyfunction]
#[pyo3(signature = (hook=None))]
pub fn _set_cfg_pass_hook(hook: Option<Py<PyAny>>) {
    *CFG_HOOK.lock().unwrap() = hook;
}

// Callables a region trace records as one graph node calling them, rather
// than tracing what they do (see torch.vm.Cfg.merge_across), by address.
static OPAQUE: Mutex<Vec<usize>> = Mutex::new(Vec::new());
static OPAQUE_ANY: AtomicBool = AtomicBool::new(false);

/// Record calls of `f` in region traces as one node calling it (`on`), or
/// stop.
#[pyfunction]
pub fn _cfg_opaque(f: &Bound<'_, PyAny>, on: bool) {
    let mut o = OPAQUE.lock().unwrap();
    let p = f.as_ptr() as usize;
    o.retain(|x| *x != p);
    if on {
        o.push(p);
    }
    OPAQUE_ANY.store(!o.is_empty(), Ordering::Relaxed);
}

#[inline]
pub(crate) fn opaque(f: *mut pyo3::ffi::PyObject) -> bool {
    OPAQUE_ANY.load(Ordering::Relaxed) && OPAQUE.lock().unwrap().contains(&(f as usize))
}

// Call sites a region trace records as one node calling their callee, by
// (code object address, code unit): only that call, not every call of the
// same callee.
static OPAQUE_SITES: Mutex<Vec<(usize, u32)>> = Mutex::new(Vec::new());
static OPAQUE_SITES_ANY: AtomicBool = AtomicBool::new(false);

/// Record the call at code unit `origin` of code object `code` (the `code`
/// and an instr's `origin` in a cfg view) in region traces as one node
/// calling its callee (`on`), or stop.
#[pyfunction]
pub fn _cfg_opaque_site(code: usize, origin: u32, on: bool) {
    let mut o = OPAQUE_SITES.lock().unwrap();
    o.retain(|x| *x != (code, origin));
    if on {
        o.push((code, origin));
    }
    OPAQUE_SITES_ANY.store(!o.is_empty(), Ordering::Relaxed);
}

#[inline]
pub(crate) fn opaque_site(code: usize, origin: u32) -> bool {
    OPAQUE_SITES_ANY.load(Ordering::Relaxed) && OPAQUE_SITES.lock().unwrap().contains(&(code, origin))
}

// What a cfg pass sees of c's Cfg as observed (see torch.vm.Cfg): its
// blocks, their instrs (op, args, line, effects, type, callee, whether
// movable) and the spans found.
fn cfg_view<'py>(py: Python<'py>, c: &Compiled, prep: &Prepared, movable: &HashSet<ValueId>, found: &[spans::Found]) -> PyResult<Bound<'py, PyDict>> {
    use pyo3::types::PyList;
    let cfg = &prep.cfg;
    let effects = unsafe { &*prep.effects.get() };
    let types = unsafe { &*prep.types.get() };
    let callees = unsafe { &*prep.callees.get() };
    let view = PyDict::new(py);
    view.set_item("qualname", &c.qualname)?;
    view.set_item("filename", &c.filename)?;
    view.set_item("code", c.code)?;
    view.set_item("path", c.path.clone())?;
    let blocks = PyList::empty(py);
    let instrs = PyDict::new(py);
    for b in cfg.block_ids() {
        let blk = &cfg[b];
        if blk.removed {
            continue;
        }
        let body: Vec<ValueId> = blk.versions.first().map_or(vec![], |v| v.body.clone());
        let seen = body.iter().any(|v| types.get(v.idx()).is_some_and(|t| t.ty.is_some()));
        let d = PyDict::new(py);
        d.set_item("id", b.0)?;
        d.set_item("params", cfg.explicit_params(b).iter().map(|p| p.0).collect::<Vec<_>>())?;
        d.set_item("instrs", body.iter().map(|v| v.0).collect::<Vec<_>>())?;
        d.set_item("succs", blk.term.successors().iter().map(|s| s.0).collect::<Vec<_>>())?;
        let term = match &blk.term {
            Terminator::Jump(Exit::Return(_)) => "return",
            Terminator::Jump(_) => "jump",
            Terminator::Branch { .. } => "branch",
            Terminator::Raise { .. } | Terminator::Reraise { .. } => "raise",
            Terminator::Unreachable => "unreachable",
        };
        d.set_item("term", term)?;
        d.set_item("seen", seen || body.is_empty())?;
        blocks.append(d)?;
        for v in &body {
            let ins = &cfg[*v];
            let i = PyDict::new(py);
            i.set_item("block", b.0)?;
            i.set_item("op", cfg.op_text(*v))?;
            i.set_item("kind", format!("{:?}", ins.op).split([' ', '(', '{']).next().unwrap_or(""))?;
            i.set_item("args", ins.args.iter().map(|a| a.0).collect::<Vec<_>>())?;
            i.set_item("line", c.line(ins.origin))?;
            i.set_item("origin", ins.origin)?;
            i.set_item("movable", movable.contains(v))?;
            i.set_item("why", IMPURE_NOW.with_borrow(|m| m.get(&(prep as *const Prepared as usize, v.0)).cloned()))?;
            let mut seq: Vec<String> = vec![];
            for (visit, _) in effects.get(Site::Instr(*v)) {
                for e in visit {
                    let e = e.to_string();
                    if !seq.contains(&e) {
                        seq.push(e);
                    }
                }
            }
            i.set_item("effects", seq)?;
            i.set_item("varies", effects.varies(Site::Instr(*v)))?;
            i.set_item("type", types.get(v.idx()).and_then(|t| t.ty.as_ref()).map(|t| t.to_string()))?;
            i.set_item("callee", callees.get(v).and_then(|f| f.get(py)))?;
            instrs.set_item(v.0, i)?;
        }
    }
    view.set_item("blocks", blocks)?;
    view.set_item("instrs", instrs)?;
    let spans = PyList::empty(py);
    for f in found {
        spans.append((f.entry.0, f.exit.map(|x| x.0)))?;
    }
    view.set_item("spans", spans)?;
    Ok(view)
}

// The edits a cfg pass returned: instrs to make movable, spans to make
// (entry, exit) and blocks no span may hold.
#[allow(clippy::type_complexity)]
fn cfg_edits(edits: &Bound<'_, PyAny>) -> PyResult<(Vec<ValueId>, Vec<(BlockId, Option<BlockId>)>, HashSet<BlockId>)> {
    if edits.is_none() {
        return Ok((vec![], vec![], HashSet::new()));
    }
    let movable: Vec<u32> = edits.get_item("movable")?.extract()?;
    let spans: Vec<(u32, Option<u32>)> = edits.get_item("spans")?.extract()?;
    let split: Vec<u32> = edits.get_item("split")?.extract()?;
    Ok((
        movable.into_iter().map(ValueId).collect(),
        spans.into_iter().map(|(e, x)| (BlockId(e), x.map(BlockId))).collect(),
        split.into_iter().map(BlockId).collect(),
    ))
}

/// Find spans when optimizing (see cfg::ir::Span): stretches of blocks, across
/// branches and loops, run as one region.
#[pyfunction]
pub fn _set_cfg_spans(on: bool) {
    SPANS.store(on, Ordering::Relaxed);
}

/// Treat calls of pure Python code as movable when optimizing (see
/// pure_compiled), so pure blocks (and regions) span them.
#[pyfunction]
pub fn _set_cfg_pure_calls(on: bool) {
    PURE_CALLS.store(on, Ordering::Relaxed);
}

// Callables whose calls are taken as pure whatever they were seen doing (see
// vm.passes.PureCalls's `pure`). Held, so that an address stays its callable's.
static USER_PURE: Mutex<Vec<Py<PyAny>>> = Mutex::new(Vec::new());
static USER_PURE_ANY: AtomicBool = AtomicBool::new(false);

/// Take every call of these callables (functions, builtins, classes) as one
/// that only computes: movable, and pure for the callers of the code it is
/// in, without looking at what its runs did.
#[pyfunction]
pub fn _set_cfg_pure_callables(fs: Vec<Py<PyAny>>) {
    USER_PURE_ANY.store(!fs.is_empty(), Ordering::Relaxed);
    *USER_PURE.lock().unwrap() = fs;
}

fn user_pure(f: *mut pyo3::ffi::PyObject) -> bool {
    USER_PURE_ANY.load(Ordering::Relaxed) && USER_PURE.lock().unwrap().iter().any(|x| x.as_ptr() == f)
}

// Whether reading a CPU tensor's value (.item() of a step count) may sit in a
// region: torch.vm's traces make the value a graph node, and refuse a trace
// that branches on it.
static SCALAR_READS: AtomicBool = AtomicBool::new(false);

/// Let reads of CPU tensors' values sit in regions (see impure_op).
#[pyfunction]
pub fn _set_cfg_scalar_reads(on: bool) {
    SCALAR_READS.store(on, Ordering::Relaxed);
}

/// A tree arg a region reads (exact lists, tuples and dicts, and registered
/// pytree nodes, of tensors and immutable values), checked as the guard
/// checks a tensor arg: the same structure, dict keys and pytree contexts,
/// each tensor's exact type, dtype, device, sizes, strides and
/// requires_grad, and each value's exact type and value.
#[pyclass(frozen)]
pub struct TreeChecks {
    root: TreeNode,
    types: Vec<Py<pyo3::types::PyType>>,
    specs: Vec<(TensorSpec, bool)>,
}

enum TreeNode {
    List(Vec<TreeNode>),
    Tuple(Vec<TreeNode>),
    Dict(Vec<(Py<PyAny>, TreeNode)>),
    Tensor(usize),
    // An int, bool, str, bytes, None or torch.Size: by exact type and value.
    Value(Py<PyAny>),
    // A registered pytree node of exactly this type, flattened by its
    // flatten_fn to children and a context equal to this one.
    Pytree { ty: Py<pyo3::types::PyType>, flatten: Py<PyAny>, context: Py<PyAny>, kids: Vec<TreeNode> },
}

impl TreeChecks {
    fn build(&mut self, x: &Bound<'_, PyAny>) -> PyResult<TreeNode> {
        use pyo3::types::{PyList, PyTuple};
        if let Ok(l) = x.cast_exact::<PyList>() {
            return Ok(TreeNode::List(l.iter().map(|y| self.build(&y)).collect::<PyResult<_>>()?));
        }
        if let Ok(t) = x.cast_exact::<PyTuple>() {
            return Ok(TreeNode::Tuple(t.iter().map(|y| self.build(&y)).collect::<PyResult<_>>()?));
        }
        if let Ok(d) = x.cast_exact::<PyDict>() {
            return Ok(TreeNode::Dict(d.iter().map(|(k, v)| Ok((k.unbind(), self.build(&v)?))).collect::<PyResult<_>>()?));
        }
        if x.is_none() || is_value_type(x) {
            return Ok(TreeNode::Value(x.clone().unbind()));
        }
        if x.extract::<&crate::bindings::ffi::Tensor>().is_err() {
            let py = x.py();
            let nodes = py.import("torch.utils._pytree")?.getattr("SUPPORTED_NODES")?;
            if let Some(node) = nodes.cast::<PyDict>()?.get_item(x.get_type())? {
                let flatten = node.getattr("flatten_fn")?;
                let (children, context): (Bound<'_, PyAny>, Bound<'_, PyAny>) = flatten.call1((x,))?.extract()?;
                let kids = children.try_iter()?.map(|y| self.build(&y?)).collect::<PyResult<_>>()?;
                return Ok(TreeNode::Pytree { ty: x.get_type().unbind(), flatten: flatten.unbind(), context: context.unbind(), kids });
            }
        }
        let t: &crate::bindings::ffi::Tensor = x.extract()?;
        let mut m = crate::bindings::ffi::TensorMeta::default();
        crate::bindings::ffi::tensor_meta(t, &mut m);
        self.types.push(x.get_type().unbind());
        let spec = TensorSpec { dtype: m.dtype, device_type: m.device_type, device_index: m.device_index, sizes: t.sizes().to_vec(), strides: t.strides().to_vec() };
        self.specs.push((spec, t.requires_grad()));
        Ok(TreeNode::Tensor(self.specs.len() - 1))
    }

    fn walk<'py>(&self, node: &TreeNode, x: &Bound<'py, PyAny>, out: &mut Vec<Bound<'py, PyAny>>) -> bool {
        use pyo3::types::{PyList, PyTuple};
        match node {
            TreeNode::List(kids) => x.cast_exact::<PyList>().is_ok_and(|l| l.len() == kids.len() && l.iter().zip(kids).all(|(y, k)| self.walk(k, &y, out))),
            TreeNode::Tuple(kids) => x.cast_exact::<PyTuple>().is_ok_and(|t| t.len() == kids.len() && t.iter().zip(kids).all(|(y, k)| self.walk(k, &y, out))),
            TreeNode::Dict(kids) => x.cast_exact::<PyDict>().is_ok_and(|d| {
                d.len() == kids.len() && d.iter().zip(kids).all(|((k, v), (want, kid))| (k.is(want) || k.eq(want).unwrap_or(false)) && self.walk(kid, &v, out))
            }),
            TreeNode::Tensor(i) => {
                let (m, rg) = &self.specs[*i];
                let ok = x.get_type().is(&self.types[*i])
                    && unsafe { crate::bindings::ffi::tensor_by_type(x.as_ptr() as usize).as_ref() }.is_some_and(|t| {
                        t.requires_grad() == *rg && crate::bindings::ffi::tensor_matches(t, m.dtype, m.device_type, m.device_index, &m.sizes, &m.strides)
                    });
                if ok {
                    out.push(x.clone());
                }
                ok
            }
            TreeNode::Value(want) => x.get_type().is(want.bind(x.py()).get_type()) && (x.is(want) || x.eq(want).unwrap_or(false)),
            TreeNode::Pytree { ty, flatten, context, kids } => {
                x.get_type().is(ty)
                    && flatten.bind(x.py()).call1((x,)).and_then(|r| r.extract::<(Bound<'py, PyAny>, Bound<'py, PyAny>)>()).is_ok_and(|(children, ctx)| {
                        ctx.eq(context).unwrap_or(false)
                            && children.len().is_ok_and(|n| n == kids.len())
                            && children.try_iter().is_ok_and(|it| it.zip(kids).all(|(y, k)| y.is_ok_and(|y| self.walk(k, &y, out))))
                    })
            }
        }
    }
}

// Whether x's type is exactly one TreeNode::Value takes (not a subclass).
fn is_value_type(x: &Bound<'_, PyAny>) -> bool {
    use pyo3::types::{PyBool, PyBytes, PyInt, PyString};
    x.is_exact_instance_of::<PyInt>()
        || x.is_exact_instance_of::<PyBool>()
        || x.is_exact_instance_of::<PyString>()
        || x.is_exact_instance_of::<PyBytes>()
        || x.get_type().qualname().is_ok_and(|q| q == "Size") && x.get_type().module().is_ok_and(|m| m == "torch")
}

#[pymethods]
impl TreeChecks {
    #[new]
    fn new(example: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut c = TreeChecks { root: TreeNode::Tuple(vec![]), types: vec![], specs: vec![] };
        c.root = c.build(example)?;
        Ok(c)
    }

    /// Whether matching runs no Python code: no registered pytree node (its
    /// flatten function is Python), and no dict key that isn't a plain
    /// value. Such a check can be asked before a region is entered (see
    /// region.rs's Quick).
    #[getter]
    fn quick(&self, py: Python<'_>) -> bool {
        fn plain(py: Python<'_>, node: &TreeNode) -> bool {
            match node {
                TreeNode::List(kids) | TreeNode::Tuple(kids) => kids.iter().all(|k| plain(py, k)),
                TreeNode::Dict(kids) => kids.iter().all(|(key, k)| (key.bind(py).is_none() || is_value_type(key.bind(py))) && plain(py, k)),
                TreeNode::Tensor(_) | TreeNode::Value(_) => true,
                TreeNode::Pytree { .. } => false,
            }
        }
        plain(py, &self.root)
    }

    /// The tensors in `x`, in order, if it matches; else None.
    fn leaves<'py>(&self, py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, pyo3::types::PyList>>> {
        let mut out = Vec::with_capacity(self.specs.len());
        if !self.walk(&self.root, x, &mut out) {
            return Ok(None);
        }
        Ok(Some(pyo3::types::PyList::new(py, out)?))
    }
}

impl TreeChecks {
    // Whether `x` matches (see leaves).
    pub(crate) fn passes(&self, x: &Bound<'_, PyAny>) -> bool {
        self.walk(&self.root, x, &mut Vec::with_capacity(self.specs.len()))
    }
}

// Torch ops a pure callee mustn't run: they read a tensor's value into
// Python (so the path taken may depend on it; with SCALAR_READS, only a
// GPU tensor's, which also syncs), or are collectives. `args` as recorded:
// a tensor not on the CPU shows its device after an @.
fn impure_op(op: &str, args: &str) -> bool {
    if SCALAR_READS.load(Ordering::Relaxed) && matches!(op, "aten::_local_scalar_dense" | "aten::item") && !args.contains('@') {
        return false;
    }
    matches!(
        op,
        "aten::_local_scalar_dense" | "aten::is_nonzero" | "aten::item" | "aten::equal" | "aten::allclose" | "aten::_assert_async" | "aten::_assert_async.msg"
    ) || op.starts_with("c10d::")
        || op.starts_with("_c10d_functional::")
        || op.starts_with("aten::_local_scalar_dense")
        || op.starts_with("aten::is_nonzero")
}

// Whether instr v of `body` writes to something that outlives the code it is
// in: an attribute, a global, a name, a cell, or an item of a container that
// `body` didn't build. A region replays tensor ops and what its trace's
// outputs hold, nothing else, so a block with such a write must not become
// one: the write would be made on the traced call alone.
pub(crate) fn escaping_write(cfg: &crate::cfg::ir::Cfg, body: &[ValueId], v: ValueId) -> bool {
    let ins = &cfg[v];
    if !writes(&ins.op) {
        return false;
    }
    let built = |x: Option<&ValueId>| {
        x.is_some_and(|x| body.contains(x) && matches!(cfg[*x].op, Op::Bytecode { opcode: opcodes::BUILD_LIST | opcodes::BUILD_MAP | opcodes::BUILD_SET, .. }))
    };
    match ins.op {
        Op::Bytecode { opcode: opcodes::STORE_SUBSCR | opcodes::STORE_SLICE, .. } => !built(ins.args.get(1)),
        Op::Bytecode { opcode: opcodes::DELETE_SUBSCR, .. } => !built(ins.args.first()),
        _ => true,
    }
}

// As escaping_write, for a span, which may change the containers it makes
// (see spans.rs): a write to an attribute, a global or a name.
pub(crate) fn escaping_store(op: &Op) -> bool {
    matches!(op, Op::StoreAttr(_))
        || matches!(
            op,
            Op::Bytecode {
                opcode: opcodes::STORE_GLOBAL | opcodes::DELETE_GLOBAL | opcodes::DELETE_ATTR | opcodes::STORE_NAME | opcodes::DELETE_NAME,
                ..
            }
        )
}

fn writes(op: &Op) -> bool {
    match op {
        Op::StoreAttr(_) | Op::StoreDeref { .. } | Op::DeleteDeref { .. } => true,
        Op::Bytecode { opcode, .. } => matches!(
            *opcode,
            opcodes::STORE_GLOBAL
                | opcodes::DELETE_GLOBAL
                | opcodes::STORE_SUBSCR
                | opcodes::DELETE_SUBSCR
                | opcodes::DELETE_ATTR
                | opcodes::STORE_NAME
                | opcodes::DELETE_NAME
                | opcodes::STORE_SLICE
        ),
        _ => false,
    }
}

const MAX_PURE_DEPTH: u32 = 40;

// What a frame being judged owns besides its locals: from slot `nargs` on
// (not its arguments'), what each free var's cell holds if the frame is of a
// closure its caller made, and with `fresh_self`, its first argument (an
// __init__'s self, which the type call made).
#[derive(Clone, Default, PartialEq, Eq, Hash)]
struct Own {
    nargs: usize,
    free: Vec<Free>,
    fresh_self: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Free {
    Opaque,
    // A container the defining frame made.
    Fresh,
    // The closure itself (it calls itself).
    SelfFn,
}

type Memo = HashMap<(usize, Own), Result<(), String>>;

fn nfree(cfg: &crate::cfg::ir::Cfg) -> usize {
    cfg.instrs.iter().find_map(|i| if let Op::CopyFreeVars(n) = i.op { Some(n as usize) } else { None }).unwrap_or(0)
}

// The free var `slot` is, if it is one.
fn free_index(cfg: &crate::cfg::ir::Cfg, slot: u32) -> Option<usize> {
    (slot as usize).checked_sub(cfg.slot_names.len() - nfree(cfg))
}

// The values every store to cell `slot` stores (none for an argument's).
fn cell_stores(cfg: &crate::cfg::ir::Cfg, slot: u32, own: &Own) -> Option<Vec<ValueId>> {
    if (slot as usize) < own.nargs || free_index(cfg, slot).is_some() {
        return None;
    }
    Some(cfg.instrs.iter().filter(|i| matches!(i.op, Op::StoreDeref { slot: s } if s == slot)).map(|i| i.args[0]).collect())
}

// What the cells of a generator site v resumed as `func` hold (see Own):
// those of the call in this frame that made it, found from v's args.
fn resumed_free(py: Python<'_>, prep: &Prepared, v: ValueId, func: &str, own: &Own) -> Vec<Free> {
    let cfg = &prep.cfg;
    let effects = unsafe { &*prep.effects.get() };
    let mut todo: Vec<(ValueId, u32)> = cfg[v].args.iter().map(|a| (*a, 0)).collect();
    while let Some((x, d)) = todo.pop() {
        if d > 4 || x.is_reserved() {
            continue;
        }
        match cfg[x].op {
            Op::Call { .. } => {
                let made = effects.get(Site::Instr(x)).iter().any(|(seq, _)| {
                    seq.iter().any(|e| matches!(e, effects::Effect::Frame { func: f, how: effects::How::Direct } if &**f == func))
                });
                if made {
                    return callee_free(py, prep, x, own);
                }
            }
            Op::GetIter => todo.push((cfg[x].args[0], d + 1)),
            Op::LoadLocal { slot } | Op::LoadLocalChecked { slot } => {
                for i in &cfg.instrs {
                    if matches!(i.op, Op::StoreLocal { slot: s } if s == slot) {
                        todo.push((i.args[0], d + 1));
                    }
                }
            }
            _ => {}
        }
    }
    vec![]
}

// What the cells of the callee of call v hold (see Own), if it's a closure
// the frame made (or the closure itself, calling itself).
fn callee_free(py: Python<'_>, prep: &Prepared, v: ValueId, own: &Own) -> Vec<Free> {
    let cfg = &prep.cfg;
    let ins = &cfg[v];
    if !matches!(ins.op, Op::Call { .. }) || ins.args.len() < 2 {
        return vec![];
    }
    // CALL's [NULL, callable, ...] or [callable, self, ...].
    let f = if ins.args[0] == crate::cfg::ir::NULL { ins.args[1] } else { ins.args[0] };
    let made = match cfg[f].op {
        Op::MakeFunction(_) => Some(f),
        Op::LoadDeref { slot } => {
            if let Some(i) = free_index(cfg, slot) {
                return if own.free.get(i) == Some(&Free::SelfFn) { own.free.clone() } else { vec![] };
            }
            match cell_stores(cfg, slot, own).as_deref() {
                Some(&[x]) if matches!(cfg[x].op, Op::MakeFunction(_)) => Some(x),
                _ => None,
            }
        }
        _ => None,
    };
    let Some(m) = made else { return vec![] };
    let Op::MakeFunction(flags) = cfg[m].op else { unreachable!() };
    if flags & 0x08 == 0 {
        return vec![];
    }
    let closure = cfg[m].args[(flags & 0x07).count_ones() as usize];
    if !matches!(cfg[closure].op, Op::BuildTuple(_)) {
        return vec![];
    }
    cfg[closure]
        .args
        .iter()
        .map(|c| {
            let Op::LoadLocal { slot } = cfg[*c].op else { return Free::Opaque };
            if let Some(i) = free_index(cfg, slot) {
                return if own.free.get(i) == Some(&Free::Fresh) { Free::Fresh } else { Free::Opaque };
            }
            match cell_stores(cfg, slot, own) {
                Some(xs) if !xs.is_empty() && xs.iter().all(|x| *x == m) => Free::SelfFn,
                Some(xs) if !xs.is_empty() && xs.iter().all(|x| fresh_container(py, prep, *x, own, &mut HashSet::new())) => Free::Fresh,
                _ => Free::Opaque,
            }
        })
        .collect()
}

// How many of a code's frame slots its arguments take.
fn arg_slots(code: &Bound<'_, PyAny>) -> usize {
    let n = |a: &str| code.getattr(a).and_then(|x| x.extract::<usize>()).unwrap_or(usize::MAX / 4);
    let flags = n("co_flags");
    // Not **kwargs, the last: a call makes that dict anew (see fresh_container).
    n("co_argcount") + n("co_kwonlyargcount") + usize::from(flags & 0x04 != 0)
}

// Whether value x of `prep`'s Cfg is a list, dict or set its frame made
// (a display, or a call of the type), directly, from a local slot (not an
// argument's) every store to which is one, or as a block's stack param every
// edge into it passes one for.
fn fresh_container(py: Python<'_>, prep: &Prepared, x: ValueId, own: &Own, seen: &mut HashSet<ValueId>) -> bool {
    let cfg = &prep.cfg;
    if !seen.insert(x) {
        return true;
    }
    match cfg[x].op {
        Op::Bytecode { opcode: opcodes::BUILD_LIST | opcodes::BUILD_MAP | opcodes::BUILD_SET, .. } => true,
        // A method call's self (LoadMethod's second output): the object.
        Op::Project(1) if matches!(cfg[cfg[x].args[0]].op, Op::LoadMethod(_)) => fresh_container(py, prep, cfg[cfg[x].args[0]].args[0], own, seen),
        // Stores of values never computed are on paths never taken.
        Op::LoadLocal { slot } | Op::LoadLocalChecked { slot } => {
            let types = unsafe { &*prep.types.get() };
            (slot as usize) >= own.nargs
                && cfg.instrs.iter().all(|i| match i.op {
                    Op::StoreLocal { slot: s } if s == slot && types.get(i.args[0].idx()).is_some_and(|t| t.ty.is_some()) => {
                        fresh_container(py, prep, i.args[0], own, seen)
                    }
                    _ => true,
                })
        }
        Op::LoadDeref { slot } => match free_index(cfg, slot) {
            Some(i) => own.free.get(i) == Some(&Free::Fresh),
            None => cell_stores(cfg, slot, own).is_some_and(|xs| !xs.is_empty() && xs.iter().all(|x| fresh_container(py, prep, *x, own, seen))),
        },
        Op::Param(crate::cfg::ir::ParamKind::Stack(_)) => {
            let Some((b, pos)) = cfg.block_ids().find_map(|b| {
                let explicit = cfg[b].params.iter().filter(|p| !matches!(cfg[**p].op, Op::Param(k) if k.is_implicit()));
                explicit.clone().position(|p| *p == x).map(|i| (b, i))
            }) else {
                return false;
            };
            if cfg.block_ids().any(|p| cfg[p].handlers.contains(&b)) {
                return false;
            }
            let mut edges = 0;
            for p in cfg.block_ids() {
                for exit in cfg[p].term.exits() {
                    let crate::cfg::ir::Exit::Block(e) = exit else { continue };
                    if e.target != b {
                        continue;
                    }
                    for ver in &cfg[p].versions {
                        edges += 1;
                        if !fresh_container(py, prep, ver.outputs[e.args[pos] as usize], own, seen) {
                            return false;
                        }
                    }
                }
            }
            edges > 0
        }
        Op::Call { .. } => {
            let callees = unsafe { &*prep.callees.get() };
            callees.get(&x).and_then(|c| c.get(py)).is_some_and(|f| makers().contains(&(f.as_ptr() as usize)))
        }
        _ => false,
    }
}

// Methods of builtin containers that change their self, by address.
fn mutators() -> &'static HashSet<usize> {
    static SET: OnceLock<HashSet<usize>> = OnceLock::new();
    SET.get_or_init(|| {
        addresses(c"
out = [list.append, list.extend, list.insert, list.pop, list.remove, list.clear, list.reverse, list.__setitem__,
       dict.update, dict.__setitem__, dict.setdefault, dict.pop, dict.popitem, dict.clear,
       set.add, set.update, set.discard, set.remove, set.clear]
")
    })
}

// Callables that return a new list, dict or set, by address.
fn makers() -> &'static HashSet<usize> {
    static SET: OnceLock<HashSet<usize>> = OnceLock::new();
    SET.get_or_init(|| {
        addresses(c"
out = [list, dict, set, sorted, str.split, str.rsplit, str.splitlines, bytes.split, list.copy, dict.copy, set.copy,
       vars(dict)['fromkeys'], set.union, set.intersection, set.difference, set.symmetric_difference]
")
    })
}

// The addresses of the objects Python `code` puts in list `out`.
fn addresses(code: &std::ffi::CStr) -> HashSet<usize> {
    Python::attach(|py| {
        let globals = pyo3::types::PyDict::new(py);
        let mut set = HashSet::new();
        if py.run(code, Some(&globals), None).is_ok()
            && let Ok(Some(out)) = globals.get_item("out")
            && let Ok(items) = out.try_iter()
        {
            for x in items.flatten() {
                set.insert(x.as_ptr() as usize);
            }
        }
        set
    })
}

// Builtins and methods of builtin types that only compute a result from
// their args (no state of their own or anyone else's changes), by address.
fn pure_builtins() -> &'static HashSet<usize> {
    static SET: OnceLock<HashSet<usize>> = OnceLock::new();
    SET.get_or_init(|| {
        Python::attach(|py| {
            let mut set = HashSet::new();
            let code = c"
import builtins, torch, math, operator
out = [getattr(builtins, n) for n in (
    'len', 'isinstance', 'issubclass', 'getattr', 'hasattr', 'type', 'tuple', 'list', 'dict', 'set',
    'frozenset', 'int', 'float', 'bool', 'str', 'min', 'max', 'abs', 'sum', 'range', 'zip', 'enumerate',
    'reversed', 'sorted', 'map', 'filter', 'any', 'all', 'iter', 'id', 'hash', 'divmod', 'round',
    'repr', 'callable', 'slice', 'pow')]
out += [dict.get, dict.items, dict.keys, dict.values, dict.__getitem__, dict.__contains__, dict.copy,
        list.__getitem__, list.copy, list.index, list.count, tuple.index, tuple.count,
        str.format, str.join, str.split, str.startswith, str.endswith, str.replace, str.lower,
        str.upper, str.strip, str.rsplit, str.partition, str.rpartition]
out += [getattr(math, n) for n in ('sqrt', 'log', 'exp', 'ceil', 'floor', 'prod', 'isfinite', 'isnan', 'log2', 'pow')]
out += [getattr(operator, n) for n in ('getitem', 'add', 'mul', 'sub', 'truediv', 'floordiv', 'index', 'neg')]
import _abc, types, bisect, itertools, functools
# Every method of the immutable builtin types, and the reads of the others.
for t in (str, bytes, tuple, frozenset, int, float, complex, bool, range, slice):
    out += [v for k, v in vars(t).items() if callable(v) and k not in ('__new__', '__init__', '__init_subclass__', '__subclasshook__', '__getattribute__', '__setattr__', '__delattr__', '__reduce__', '__reduce_ex__', '__sizeof__', '__dir__')]
out += [list.__len__, list.__contains__, list.__iter__, list.__reversed__, list.__add__, list.__mul__, list.__eq__,
        dict.__len__, dict.__iter__, dict.__or__, dict.__eq__,
        set.union, set.intersection, set.difference, set.symmetric_difference, set.issubset, set.issuperset,
        set.isdisjoint, set.copy, set.__contains__, set.__len__, set.__iter__, set.__or__, set.__and__, set.__sub__, set.__xor__]
out += [getattr(itertools, n) for n in ('chain', 'combinations', 'combinations_with_replacement', 'permutations', 'product',
        'islice', 'accumulate', 'repeat', 'zip_longest', 'starmap', 'compress', 'takewhile', 'dropwhile', 'filterfalse')]
import decimal, fractions
out += [vars(itertools.chain)['from_iterable'], vars(dict)['fromkeys'], bisect.bisect_left, bisect.bisect_right, functools.partial, decimal.Decimal, fractions.Fraction]
out += [getattr(builtins, n) for n in ('ord', 'chr', 'format', 'bin', 'hex', 'oct')]
out += [getattr(math, n) for n in ('gcd', 'lcm', 'comb', 'perm', 'factorial', 'log10', 'fabs', 'isinf', 'trunc', 'fsum')]
out += [getattr(operator, n) for n in ('itemgetter', 'attrgetter', 'eq', 'ne', 'lt', 'le', 'gt', 'ge', 'not_', 'truth',
        'contains', 'mod', 'and_', 'or_', 'xor', 'invert', 'lshift', 'rshift', 'pos', 'abs')]
out += [_abc._abc_instancecheck, _abc._abc_subclasscheck, types.ModuleType.__getattribute__]
out += [getattr(torch._C, n) for n in (
    '_get_native_aot_enabled', '_is_tracing', '_get_tracing_state', '_has_torch_function',
    '_has_torch_function_unary', '_has_torch_function_variadic', '_is_torch_function_enabled',
    '_is_torch_function_mode_enabled', '_get_cudnn_enabled', '_get_privateuse1_backend_name',
    '_are_functorch_transforms_active') if hasattr(torch._C, n)]
out += [torch.is_tensor, torch.is_floating_point, torch.is_grad_enabled, torch.is_autocast_enabled,
        torch.get_default_dtype, torch.Size, torch.device, torch._C._get_tracing_state,
        torch.Tensor.size, torch.Tensor.dim, torch.Tensor.numel, torch.Tensor.stride,
        torch.Tensor.is_contiguous, torch.Tensor.element_size, torch.Tensor.get_device]
";
            let globals = pyo3::types::PyDict::new(py);
            if py.run(code, Some(&globals), None).is_ok()
                && let Ok(Some(out)) = globals.get_item("out")
                && let Ok(items) = out.try_iter()
            {
                for x in items.flatten() {
                    set.insert(x.as_ptr() as usize);
                }
            }
            set
        })
    })
}

// Whether instr v of `prep`'s Cfg, as observed, only computed (Err: why
// not): it writes nothing, every visit ran only torch ops pure code may run
// and frames of its pure clone, and a call that entered no frame and ran no
// torch op called a pure builtin or an in-graph torch function. memo: by
// Compiled address and what the frame owns.
fn pure_instr(py: Python<'_>, prep: &Prepared, v: ValueId, clones: &HashMap<ValueId, SiteClone>, depth: u32, own: &Own, memo: &mut Memo) -> Result<(), String> {
    let ins = &prep.cfg[v];
    let here = || format!("%{} {:?}", v.0, ins.op);
    // In a callee (depth > 0), its own cells, and containers it made, die
    // with its frame (unless it returns them, as a region rebuilds them).
    let own_cell = |slot: u32| free_index(&prep.cfg, slot).is_none();
    let is_self = |x: ValueId| {
        matches!(prep.cfg[x].op, Op::LoadLocal { slot: 0 } | Op::LoadLocalChecked { slot: 0 })
            && !prep.cfg.instrs.iter().any(|i| matches!(i.op, Op::StoreLocal { slot: 0 } | Op::StoreDeref { slot: 0 }))
    };
    let own_container = |x: ValueId| fresh_container(py, prep, x, own, &mut HashSet::new());
    let local = depth > 0
        && match ins.op {
            Op::StoreDeref { slot } | Op::DeleteDeref { slot } => own_cell(slot),
            Op::StoreAttr(_) => own.fresh_self && is_self(ins.args[1]),
            Op::Bytecode { opcode: opcodes::STORE_SUBSCR | opcodes::STORE_SLICE, .. } => own_container(ins.args[1]),
            Op::Bytecode {
                opcode:
                    opcodes::DELETE_SUBSCR
                    | opcodes::LIST_APPEND
                    | opcodes::SET_ADD
                    | opcodes::MAP_ADD
                    | opcodes::LIST_EXTEND
                    | opcodes::SET_UPDATE
                    | opcodes::DICT_UPDATE
                    | opcodes::DICT_MERGE,
                ..
            } => own_container(ins.args[0]),
            _ => false,
        };
    if writes(&ins.op) && !local {
        return Err(format!("{} writes", here()));
    }
    // CALL_INTRINSIC_1's unary + (5) and list-to-tuple (6) only compute.
    if let Op::Bytecode { opcode, arg } = ins.op
        && (matches!(
            opcode,
            opcodes::BEFORE_WITH | opcodes::BEFORE_ASYNC_WITH | opcodes::WITH_EXCEPT_START | opcodes::SETUP_ANNOTATIONS | opcodes::LOAD_BUILD_CLASS | opcodes::CALL_INTRINSIC_2
        ) || (opcode == opcodes::CALL_INTRINSIC_1 && arg != 5 && arg != 6))
    {
        return Err(format!("{} isn't pure", here()));
    }
    // A callee's own functions (closures over its cells) die with its frame
    // unless returned, and a region refuses to return those; a callee that's
    // a generator yields to the frame consuming it.
    if matches!(ins.op, Op::Send | Op::Region(_) | Op::Autograd(_)) || (depth == 0 && matches!(ins.op, Op::MakeFunction(_) | Op::Yield)) {
        return Err(format!("{} isn't pure", here()));
    }
    if !ins.op.can_raise() {
        return Ok(());
    }
    let callees = unsafe { &*prep.callees.get() };
    if is_call(&ins.op) && callees.get(&v).and_then(|c| c.get(py)).is_some_and(|f| user_pure(f.as_ptr())) {
        return Ok(());
    }
    let effects = unsafe { &*prep.effects.get() };
    let (mut frames, mut torch_ops, mut indirect) = (None, false, false);
    for (seq, _) in effects.get(Site::Instr(v)) {
        for e in seq {
            match e {
                effects::Effect::TorchOp { op, args } if impure_op(op, args) => return Err(format!("{} ran {op}", here())),
                effects::Effect::TorchOp { .. } => torch_ops = true,
                effects::Effect::Frame { func, how: effects::How::Direct } => frames = Some(func.clone()),
                effects::Effect::Frame { func, how: effects::How::Indirect } => {
                    pure_indirect(py, prep, v, func, depth, memo).map_err(|e| format!("{} entered {func} indirectly: {e}", here()))?;
                    indirect = true;
                }
                // A generator's frame (one the site made, if pure).
                effects::Effect::Frame { func, how: effects::How::Resume } => {
                    let code = effects::code_named(py, func).ok_or_else(|| format!("{} resumed {func}: its code is unknown", here()))?;
                    pure_code(py, &code, resumed_free(py, prep, v, func, own), depth, memo).map_err(|e| format!("{} resumed {func}: {e}", here()))?;
                }
            }
        }
    }
    if let Some(func) = frames {
        // A clone that hasn't run (its caller stopped running it), or none
        // (too deep, a generator, or a closure the caller made): judged from
        // the code's own Compiled, which runs the other frames.
        return match clones.get(&v) {
            Some(cl) if cl.compiled.auto_optimized.load(Ordering::Relaxed) => {
                let own = Own { nargs: arg_slots(cl.code.bind(py)), free: callee_free(py, prep, v, own), fresh_self: false };
                pure_compiled(py, &cl.compiled, own, depth + 1, memo).map_err(|e| format!("{} -> {}: {e}", here(), cl.compiled.qualname))
            }
            Some(cl) => pure_code(py, cl.code.bind(py), callee_free(py, prep, v, own), depth, memo).map_err(|e| format!("{} -> {} (own): {e}", here(), cl.compiled.qualname)),
            None => {
                let callees = unsafe { &*prep.callees.get() };
                let code = callees.get(&v).and_then(|c| c.get(py)).and_then(|f| f.getattr("__code__").ok()).or_else(|| effects::code_named(py, &func));
                let code = code.ok_or_else(|| format!("{} entered {func} but has no clone", here()))?;
                pure_code(py, &code, callee_free(py, prep, v, own), depth, memo).map_err(|e| format!("{} -> {func} (own): {e}", here()))
            }
        };
    }
    if !is_call(&ins.op) || torch_ops {
        return Ok(());
    }
    let Some(f) = callees.get(&v).and_then(|c| c.get(py)) else {
        let ran = unsafe { &*prep.types.get() }.get(v.idx()).is_some_and(|t| t.ty.is_some());
        let what = match callees.get(&v) {
            None if !ran => "never ran",
            None => "no callee recorded",
            Some(effects::Callee::Many) => "several callees",
            Some(_) => "a callee that's gone",
        };
        let types = unsafe { &*prep.types.get() };
        let ty = |k: usize| ins.args.get(k).and_then(|a| types.get(a.idx())).and_then(|t| t.ty.as_ref()).map_or("?".into(), |t| t.to_string());
        return Err(format!("{} called something unknown ({what}; a {} or {})", here(), ty(1), ty(0)));
    };
    // Through a functools.partial, `f` is what the partial calls, and the
    // instr's args aren't f's alone.
    let partial = callees.get(&v).and_then(|c| c.via()) == Some(effects::Via::Partial);
    if pure_builtins().contains(&(f.as_ptr() as usize)) || is_in_graph(&f) {
        Ok(())
    } else if partial && indirect && f.is_instance_of::<pyo3::types::PyFunction>() {
        // A Python function, whose frames were judged above.
        Ok(())
    } else if plain_class(py, &f) {
        // A new object, whose __init__ (if Python's) entered a frame judged above.
        Ok(())
    } else if unsafe { effects::lru_cache_wrapper(f.as_ptr()) } {
        // A hit (a miss enters the function's frame, judged above).
        Ok(())
    } else if depth > 0 && !partial && mutators().contains(&(f.as_ptr() as usize)) && ins.args.len() > 1 && own_container(ins.args[1]) {
        // A method changing a container this frame made (CALL's [method, self, ...]).
        Ok(())
    } else {
        Err(format!("{} called {}", here(), f.repr().map_or("?".into(), |r| r.to_string())))
    }
}

thread_local! {
    // The code being optimized, for TORCH_RUST_PURE_DEBUG, and its lines.
    static PURE_DEBUG_CODE: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
    static PURE_DEBUG_LINES: std::cell::RefCell<Vec<(u32, u32, i32)>> = const { std::cell::RefCell::new(Vec::new()) };
}

thread_local! {
    // Why the sites judged in the optimizing run in progress aren't pure,
    // by (Prepared, instr): what the view of a cfg pass says (see cfg_view).
    static IMPURE_NOW: std::cell::RefCell<HashMap<(usize, u32), String>> = std::cell::RefCell::new(HashMap::new());
}

// Why a site isn't pure (see pure_site), as (code, line, why): what
// TORCH_RUST_PURE_DEBUG prints, kept for torch.vm.regions. Each once.
static IMPURE: Mutex<Vec<(String, i32, String)>> = Mutex::new(Vec::new());
const MAX_IMPURE: usize = 4096;

/// Why the sites that keep the code around them out of a region aren't pure:
/// (qualname@depth, line, why) for each call that isn't movable and each
/// other instr with effects that isn't, as judged when their code was last
/// optimized.
#[pyfunction]
pub fn _cfg_impure() -> Vec<(String, i32, String)> {
    IMPURE.lock().unwrap().clone()
}

// Whether the frames of `code` were only seen computing, judged from its own
// Compiled (which must have run).
fn pure_code(py: Python<'_>, code: &Bound<'_, PyAny>, free: Vec<Free>, depth: u32, memo: &mut Memo) -> Result<(), String> {
    pure_code_as(py, code, free, false, depth, memo)
}

fn pure_code_as(py: Python<'_>, code: &Bound<'_, PyAny>, free: Vec<Free>, fresh_self: bool, depth: u32, memo: &mut Memo) -> Result<(), String> {
    let c = unsafe { lookup(code) }.map_err(|e| e.to_string())?;
    if !c.auto_optimized.load(Ordering::Relaxed) {
        return Err("it hasn't run".into());
    }
    pure_compiled(py, c, Own { nargs: arg_slots(code), free, fresh_self }, depth + 1, memo)
}

// Whether a function a C caller (a type slot like __getitem__ or __iter__,
// a builtin) ran for site v was only seen computing: judged from its code's
// own Compiled, which every such frame runs. An autograd.Function's forward
// isn't (a region would lose its backward).
fn pure_indirect(py: Python<'_>, prep: &Prepared, v: ValueId, func: &str, depth: u32, memo: &mut Memo) -> Result<(), String> {
    let callees = unsafe { &*prep.callees.get() };
    // torch._native's Python kernels, which the dispatcher runs for an aten
    // op the site called: part of the op.
    if func.starts_with("torch._native.") {
        return Ok(());
    }
    let code = effects::code_named(py, func).ok_or("its code is unknown")?;
    let callee = callees.get(&v).and_then(|c| c.get(py));
    // The __init__ a class call ran, on the object its __new__ (object's)
    // just made; or an autograd.Function's forward(ctx, ...), on the ctx its
    // apply just made (a region traces the apply whole, with its backward).
    let fresh_self = if callee.as_ref().is_some_and(|f| is_function_apply(py, f)) {
        let ctx_first = code.getattr("co_varnames").and_then(|n| n.get_item(0)).is_ok_and(|n| n.to_string() == "ctx");
        if !func.ends_with(".forward") || !ctx_first {
            return Err("an autograd.Function's".into());
        }
        true
    } else {
        func.ends_with(".__init__") && callee.is_some_and(|f| plain_class(py, &f))
    };
    pure_code_as(py, &code, vec![], fresh_self, depth, memo)
}

// Whether `f` is torch._C._FunctionBase.apply, which runs an autograd.Function.
fn is_function_apply(py: Python<'_>, f: &Bound<'_, PyAny>) -> bool {
    static APPLY: OnceLock<usize> = OnceLock::new();
    let apply = *APPLY.get_or_init(|| {
        py.import("torch._C")
            .and_then(|m| m.getattr("_FunctionBase"))
            .and_then(|b| b.getattr("__dict__"))
            .and_then(|d| d.get_item("apply"))
            .map_or(0, |a| a.as_ptr() as usize)
    });
    f.as_ptr() as usize == apply
}

// Whether `f` is a class whose objects object.__new__ makes and object's or
// a Python function's __init__ initializes.
fn plain_class(py: Python<'_>, f: &Bound<'_, PyAny>) -> bool {
    let object = py.get_type::<pyo3::types::PyAny>();
    let same = |name: &str| f.getattr(name).is_ok_and(|m| object.getattr(name).is_ok_and(|n| m.is(&n)));
    f.is_instance_of::<pyo3::types::PyType>() && same("__new__") && (same("__init__") || f.getattr("__init__").is_ok_and(|i| i.is_instance_of::<pyo3::types::PyFunction>()))
}

// The torch ops instr v of `prep` ran on its busiest visit: one per op it
// ran itself and per frame it entered, and for a frame of its clone the ops
// that clone's frames ran (at least one).
pub(crate) fn site_ops(prep: &Prepared, v: ValueId, clones: &HashMap<ValueId, SiteClone>, depth: u32) -> usize {
    let effects = unsafe { &*prep.effects.get() };
    let of = |e: &effects::Effect| match e {
        effects::Effect::Frame { how: effects::How::Direct, .. } if depth < MAX_PURE_DEPTH => clones.get(&v).map_or(1, |cl| ops_ran(&cl.compiled, depth + 1).max(1)),
        _ => 1,
    };
    effects.get(Site::Instr(v)).iter().map(|(seq, _)| seq.iter().map(of).sum::<usize>()).max().unwrap_or(0)
}

// The torch ops the frames of `c` ran, as its first Plan observed them.
fn ops_ran(c: &Compiled, depth: u32) -> usize {
    let first = unsafe { &*(&*c.plans.lock().unwrap()[0] as *const Plan) };
    let clones = &c.plan().prep.clones;
    let cfg = &first.prep.cfg;
    cfg.block_ids().filter_map(|b| cfg[b].versions.first()).flat_map(|ver| ver.body.iter()).map(|v| site_ops(&first.prep, *v, clones, depth)).sum()
}

// Pure blocks that aren't traced, and why (see Act::region_worthy), as
// (name, state, why): for torch.vm.regions. Each once.
static UNTRACED: Mutex<Vec<(String, String, String)>> = Mutex::new(Vec::new());

pub(crate) fn note_untraced(name: String, state: &str, why: String) {
    if std::env::var("TORCH_RUST_REGION_DEBUG").is_ok_and(|d| d == "2") {
        eprintln!("region {name}: not traced: {why}");
    }
    let mut seen = UNTRACED.lock().unwrap();
    if seen.len() < MAX_IMPURE && !seen.iter().any(|(n, _, _)| *n == name) {
        seen.push((name, state.to_string(), why));
    }
}

/// The blocks that only compute and still aren't traced into a region, as
/// (name, state, why): too little work, or a store that outlives the block.
#[pyfunction]
pub fn _cfg_untraced() -> Vec<(String, String, String)> {
    UNTRACED.lock().unwrap().clone()
}

fn is_call(op: &Op) -> bool {
    matches!(op, Op::Call { .. } | Op::Bytecode { opcode: opcodes::CALL_FUNCTION_EX, .. })
}

// Whether a site of the caller (instr v of `prep`) is pure (see pure_instr).
fn pure_site(py: Python<'_>, prep: &Prepared, v: ValueId, clones: &HashMap<ValueId, SiteClone>, memo: &mut Memo) -> bool {
    match pure_instr(py, prep, v, clones, 0, &Own::default(), memo) {
        Ok(()) => true,
        Err(why) => {
            let effects = !effects_empty(prep, v);
            if std::env::var_os("TORCH_RUST_PURE_DEBUG").is_some() && effects {
                PURE_DEBUG_CODE.with_borrow(|q| eprintln!("impure: {q}: {why}"));
            }
            let ins = &prep.cfg[v];
            let call = is_call(&ins.op);
            IMPURE_NOW.with_borrow_mut(|m| m.insert((prep as *const Prepared as usize, v.0), why.clone()));
            if effects || call {
                let code = PURE_DEBUG_CODE.with_borrow(|q| q.clone());
                let line = PURE_DEBUG_LINES.with_borrow(|l| l.iter().find(|(s, e, _)| *s <= ins.origin && ins.origin < *e).map_or(0, |x| x.2));
                let mut seen = IMPURE.lock().unwrap();
                if seen.len() < MAX_IMPURE && !seen.iter().any(|(c, l, w)| *c == code && *l == line && *w == why) {
                    seen.push((code, line, why));
                }
            }
            false
        }
    }
}

fn effects_empty(prep: &Prepared, v: ValueId) -> bool {
    unsafe { &*prep.effects.get() }.get(Site::Instr(v)).iter().all(|(seq, _)| seq.is_empty())
}

// Whether the code a Compiled runs (a clone: one call site's frames) was only
// seen computing (see pure_instr): every instr of the blocks it ran.
fn pure_compiled(py: Python<'_>, c: &Compiled, own: Own, depth: u32, memo: &mut Memo) -> Result<(), String> {
    let key = (c as *const Compiled as usize, own.clone());
    if let Some(r) = memo.get(&key) {
        return r.clone();
    }
    if depth > MAX_PURE_DEPTH {
        return Err("too deep".into());
    }
    // Assume pure while looking (recursion through the same Compiled).
    memo.insert(key.clone(), Ok(()));
    let first = unsafe { &*(&*c.plans.lock().unwrap()[0] as *const Plan) };
    let clones = &c.plan().prep.clones;
    let prep = &first.prep;
    let cfg = &prep.cfg;
    let types = unsafe { &*prep.types.get() };
    let effects = unsafe { &*prep.effects.get() };
    let ran = |v: &ValueId| types.get(v.idx()).is_some_and(|t| t.ty.is_some());
    // A call a cfg pass made movable in this code (Cfg.merge_across) is
    // taken as the pass said, here as in the code's own blocks.
    let declared = c.declared.lock().unwrap().clone();
    let said = |v: &ValueId| is_call(&cfg[*v].op) && declared.contains(&cfg[*v].origin);
    let r = cfg.block_ids().try_for_each(|b| {
        let blk = &cfg[b];
        let body = blk.versions.first().map_or(&[][..], |ver| &ver.body[..]);
        let entered = b == crate::cfg::ir::ENTRY || blk.params.iter().any(ran) || body.iter().any(ran);
        if !entered {
            return Ok(());
        }
        body.iter().filter(|v| !said(v)).try_for_each(|v| pure_instr(py, prep, *v, clones, depth, &own, memo))?;
        let ok = effects.get(Site::Block(b)).iter().all(|(seq, _)| seq.iter().all(|e| matches!(e, effects::Effect::TorchOp { op, args } if !impure_op(op, args))));
        if ok { Ok(()) } else { Err(format!("bb{} has effects", b.0)) }
    });
    if let Err(e) = &r
        && e.ends_with("(never ran)")
        && std::env::var_os("TORCH_RUST_PURE_DEBUG").is_some()
    {
        let typed = types.iter().filter(|t| t.ty.is_some()).count();
        eprintln!(
            "never-ran: {} depth {} plans {} auto_optimized {} returns {} typed {}/{} instrs",
            c.qualname,
            c.depth,
            c.plans.lock().unwrap().len(),
            c.auto_optimized.load(Ordering::Relaxed),
            c.returns.load(Ordering::Relaxed),
            typed,
            cfg.instrs.len()
        );
    }
    memo.insert(key, r.clone());
    r
}

// Whether Python function `f` ran its torch ops in its own frame, entering
// no other Python frame, and writes nothing but its own locals: then f is a
// leaf, which Dynamo's in-graph functions (`F.relu`, `F.layer_norm`) usually
// are, and a call of it can be treated like a call of a builtin.
fn is_leaf(f: &Bound<'_, PyAny>) -> bool {
    let Ok(code) = f.getattr("__code__") else { return false };
    let Ok(c) = (unsafe { lookup(&code) }) else { return false };
    let first = unsafe { &*(&*c.plans.lock().unwrap()[0] as *const Plan) };
    let writes = first.prep.cfg.instrs.iter().any(|i| match i.op {
        Op::StoreAttr(_) | Op::StoreDeref { .. } | Op::DeleteDeref { .. } => true,
        Op::Bytecode { opcode, .. } => matches!(
            opcode,
            opcodes::STORE_GLOBAL | opcodes::DELETE_GLOBAL | opcodes::STORE_SUBSCR | opcodes::DELETE_SUBSCR | opcodes::DELETE_ATTR | opcodes::STORE_NAME | opcodes::DELETE_NAME
        ),
        _ => false,
    });
    let effects = unsafe { &*first.prep.effects.get() };
    !writes && effects.sites().all(|(seqs, varies)| !varies && seqs.iter().flat_map(|(seq, _)| seq).all(|e| matches!(e, effects::Effect::TorchOp { .. })))
}

// Whether `f` is torch._C._has_torch_function{,_unary,_variadic}.
fn is_has_torch_function(f: &Bound<'_, PyAny>) -> bool {
    let Ok(c) = f.py().import("torch._C") else { return false };
    ["_has_torch_function", "_has_torch_function_unary", "_has_torch_function_variadic"]
        .iter()
        .any(|n| c.getattr(*n).is_ok_and(|g| g.is(f)))
}

// Whether Dynamo puts calls to `f` in its graphs.
fn is_in_graph(f: &Bound<'_, PyAny>) -> bool {
    f.py()
        .import("torch._dynamo.trace_rules")
        .and_then(|m| m.call_method1("lookup", (f,)))
        .and_then(|r| r.getattr("__name__"))
        .and_then(|n| n.extract::<String>())
        .is_ok_and(|n| n == "TorchInGraphFunctionVariable")
}

// The Compiled of `code` (or of a function's `__code__`), which lives as long
// as the code object.
unsafe fn lookup<'a>(code: &Bound<'_, PyAny>) -> PyResult<&'a Compiled> {
    let code = if code.hasattr("__code__")? {
        code.getattr("__code__")?
    } else {
        code.clone()
    };
    let mut p: *mut c_void = std::ptr::null_mut();
    let index = extra_index();
    if index < 0 || PyUnstable_Code_GetExtra(code.as_ptr(), index, &mut p) < 0 {
        return Err(PyErr::fetch(code.py()));
    }
    if p.is_null() || p == trampoline::mark() {
        return Err(pyo3::exceptions::PyValueError::new_err("the CFG executor hasn't run this code"));
    }
    Ok(&*(p as *const Compiled))
}

// The Compiled of `code` (see lookup), or of the clone `path` leads to: each
// instr has a clone in the Compiled before it. Clones live as long as the
// Plans holding them, so as long as `code`'s Compiled.
unsafe fn resolve<'a>(code: &Bound<'_, PyAny>, path: &[u32]) -> PyResult<&'a Compiled> {
    let mut c: &Compiled = lookup(code)?;
    for v in path {
        let Some(cl) = c.plan().prep.clones.get(&ValueId(*v)) else {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("%{v} has no clone")));
        };
        c = &*Arc::as_ptr(&cl.compiled);
    }
    Ok(c)
}

/// For coverage reports: the clones the CFG the executor runs for `code` has
/// (or the clone `clone` leads to, see cfg_observed), as (instr, the cloned
/// function's qualname).
#[pyfunction]
#[pyo3(signature = (code, clone=vec![]))]
pub fn _cfg_clones(code: &Bound<'_, PyAny>, clone: Vec<u32>) -> PyResult<Vec<(u32, String)>> {
    let c = unsafe { resolve(code, &clone)? };
    let mut out: Vec<(u32, String)> = c
        .plan()
        .prep
        .clones
        .iter()
        .map(|(v, cl)| (v.0, cl.code.bind(code.py()).getattr("co_qualname").and_then(|q| q.extract()).unwrap_or_default()))
        .collect();
    out.sort();
    Ok(out)
}

struct Observed<'a> {
    by_cfg: HashMap<*const Cfg, &'a Prepared>,
    effects: bool,
    types: bool,
}

impl Annotations for Observed<'_> {
    fn notes(&self, cfg: &Cfg, site: Site) -> Vec<String> {
        if !self.effects {
            return vec![];
        }
        let prep = self.by_cfg[&(cfg as *const Cfg)];
        let mut lines = vec![];
        if let Site::Instr(v) = site {
            // Describing runs Python code, which may record, so not holding
            // the table.
            let py = unsafe { Python::assume_attached() };
            let callee = unsafe { &*prep.callees.get() }.get(&v).map(|c| c.clone_ref(py));
            if prep.clones.contains_key(&v) {
                lines.push("cloned".into());
            }
            if let Some(c) = callee {
                let via = match c.via() {
                    Some(effects::Via::Call) => " (via __call__)",
                    Some(effects::Via::Getattr) => " (via __getattr__)",
                    _ => "",
                };
                lines.push(format!("callee {}{via}", c.describe(py)));
            }
        }
        let table = unsafe { &*prep.effects.get() };
        for (seq, n) in table.get(site) {
            let s = if *n == 1 { "" } else { "s" };
            lines.push(format!("effects [{n} visit{s}]:"));
            lines.extend(seq.iter().map(|e| format!("  {e}")));
        }
        if table.varies(site) {
            lines.push("effects vary (more sequences than kept)".into());
        }
        lines
    }

    fn value(&self, cfg: &Cfg, v: ValueId) -> Option<String> {
        if !self.types {
            return None;
        }
        let table = unsafe { &*self.by_cfg[&(cfg as *const Cfg)].types.get() };
        table[v.idx()].ty.as_ref().map(|t| t.to_string())
    }
}

/// Call `func(*args, **kwargs)`, running every Python frame entered on this
/// thread during the call through the CFG executor.
#[pyfunction]
#[pyo3(signature = (func, /, *args, **kwargs))]
pub fn call_with_cfg_execution<'py>(
    func: &Bound<'py, PyAny>,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    trampoline::init(func.py())?;
    region::next_call();
    let _guard = ExecGuard::new();
    let _ops = (RECORD.load(Ordering::Relaxed) || torch_ops::printing()).then(torch_ops::OpHookGuard::new);
    func.call(args, kwargs)
}
