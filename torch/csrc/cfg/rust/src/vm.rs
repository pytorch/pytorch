//! The virtual machine: a register interpreter over the SSA graph.
//!
//! Each basic block runs in one of three tiers:
//!   - generic: the instructions have the semantics of the bytecode they came
//!     from and go through the CPython API. Anything Python can do runs here,
//!     so what other compilers call a graph break is just a block (or an
//!     instruction) that stays in this tier;
//!   - profiling: generic, with the dispatcher capture hook installed around
//!     the operations that may be tensor operations, and with the values that
//!     flow by being recorded. Real tensors, real kernels: the first run of a
//!     block is its trace;
//!   - specialized: a version of the block produced by the pass pipeline from
//!     the profile. It calls post-dispatch kernels directly and may depend on
//!     guards; a failed guard resumes the profiling tier, in this block only.

use crate::analysis::{self, Analysis};
use crate::bytecode::OpcodeTable;
use crate::capi::{self, SiteSlot, TcfgApi, TcfgPyOp, TcfgTensorInfo};
use crate::graph::{self, Assert, Cond, Def, Exported, GraphSite, Kind, Path, PathValue, Stage, Trace};
use crate::ir::*;
use crate::pyffi::{self, new_ref, py_bool, xdecref, Obj, PyObj, Shared};
use crate::{frontend, inline, unroll};
use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use crate::spec::{self, CalleeKind, InstFacts, Kernels, PassCx, Pipeline, Unit, ValueObs};
use pyo3::ffi;
use std::ffi::CString;
use std::os::raw::c_char;
use std::ptr::null_mut;
use std::sync::atomic::Ordering;
use std::sync::Arc;

#[derive(Clone)]
pub struct Options {
    /// Run the profiling tier and specialize. Off: generic tier only.
    pub specialize: bool,
    /// Let call sites enter below the Autograd kernel when no gradient is
    /// being tracked.
    pub skip_autograd: bool,
    pub max_versions: usize,
    /// How many times a block may go back to the profiling tier.
    pub max_respecializations: u32,
    pub max_site_variants: usize,
    /// Keep the tensors kernels return unboxed until Python code needs them.
    pub unboxed: bool,
    /// Inline calls of functions of the same module.
    pub inline: bool,
    /// Unroll loops over a small constant range, of at most this many
    /// iterations.
    pub unroll: bool,
    pub max_trip_count: i64,
    /// Compile the Python functions that compiled code calls.
    pub callees: bool,
    /// Take the registered shortcuts through library code (see shortcuts.rs).
    pub shortcuts: bool,
    /// Time each callee both compiled and interpreted, and leave it to the
    /// interpreter if that is where it runs faster.
    pub demote: bool,
    /// Compile also the Python functions that are called by C code while a
    /// compiled function runs, or by a function the interpreter runs (see
    /// `take_frame`).
    pub hook: bool,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            specialize: true,
            skip_autograd: true,
            max_versions: 4,
            max_respecializations: 6,
            max_site_variants: 4,
            unboxed: true,
            inline: true,
            unroll: true,
            max_trip_count: unroll::MAX_TRIP_COUNT,
            callees: true,
            shortcuts: true,
            demote: true,
            hook: false,
        }
    }
}

pub struct BlockState {
    pub versions: Vec<Arc<BlockCode>>,
    /// Versions that were replaced. They are kept: an activation further up
    /// the stack may still be running one, and the block loop holds plain
    /// pointers to the code it runs. At most `max_respecializations` exist.
    pub retired: Vec<Arc<BlockCode>>,
    pub slots: Vec<Option<Arc<SiteSlot>>>,
    /// Why an instruction that made dispatcher calls did not become a site.
    pub reasons: Vec<Option<String>>,
    pub capture_attempts: Vec<u32>,
    /// What the profiling tier saw each instruction do.
    pub facts: Vec<InstFacts>,
    pub respec_left: u32,
    pub profile_runs: u32,
    /// How many profiling runs in a row ended in a raise.
    pub raised: u32,
    pub next_version_id: u32,
    pub execs: u64,
    pub spec_execs: u64,
    pub entry_misses: u64,
    pub deopts: u64,
    pub rejected: Option<String>,
    /// What a pass left here for its next run on this block.
    pub note: Option<PyObj>,
    /// The graphs that start in the block, each by the generic instruction
    /// its first operation stands for: the versions of the block share
    /// them, so what was recorded for one is there for the next.
    pub graphs: Vec<(u32, Arc<GraphSite>)>,
}

pub struct State {
    pub blocks: Vec<BlockState>,
    pub obs: Vec<ValueObs>,
    pub calls: u64,
    pub log: Vec<String>,
}

/// Inline cache of a global or module attribute load, keyed on the version
/// tags of the dictionaries involved: as long as a dict's tag is unchanged
/// the dict has not been modified, so the value it holds for the name is
/// still there (and still alive, the dict owns it).
#[derive(Clone, Copy)]
struct LoadCache {
    owner: Obj,
    version: u64,
    builtins_version: u64,
    value: Obj,
}

impl LoadCache {
    const EMPTY: LoadCache =
        LoadCache { owner: null_mut(), version: u64::MAX, builtins_version: u64::MAX, value: null_mut() };
}

/// What the block loop needs to enter a block whose choice of code no
/// longer depends on anything: the code (null while the block still has to
/// be looked at), whether it is a specialized version, and how often it was
/// entered this way. Only touched with the GIL held.
#[derive(Clone, Copy)]
struct Settled {
    code: *const BlockCode,
    specialized: bool,
    entries: u64,
}

struct SettledBlocks {
    blocks: Vec<std::cell::Cell<Settled>>,
    calls: std::cell::Cell<u64>,
    /// How many calls of the function are running with its first block in
    /// the profiling tier.
    warming: std::cell::Cell<u32>,
    trial: std::cell::Cell<Trial>,
    /// For the function that was handed to torch.vm: whether calls
    /// below its last call were part of a measurement (`watched`), and
    /// whether they took a noticeable part of it (`below`).
    watched: std::cell::Cell<bool>,
    below: std::cell::Cell<bool>,
}

/// Whether a callee is worth running compiled is measured, not assumed: the
/// generic tier is slower than the interpreter on plain Python, so a callee
/// without much tensor work can lose more than it gains. After some calls to
/// warm up, calls alternate between the compiled form and the interpreter
/// and are timed, and the faster of the two (by their best times) is what
/// the callee is from then on.
///
/// A function and what it calls are measured at the same time. Compiled code
/// only gets faster when something it calls is handed to the interpreter, so
/// a function can win on any of its samples; to lose, it needs samples
/// during which nothing it calls was still being measured (`settled`): a
/// caller is not given up for the time its callees took before they were
/// where they belong.
#[derive(Clone, Copy)]
struct Trial {
    phase: u8,
    /// The value of PLAN_EPOCH the measurement started under.
    epoch: u32,
    /// The value of ENTRY_CALLS when the callee was first measured.
    born: u32,
    calls: u32,
    compiled_best: u64,
    compiled_samples: u8,
    settled_best: u64,
    settled_samples: u8,
    interpreted_best: u64,
    interpreted_samples: u8,
}

impl Trial {
    /// How many samples of each kind decide it.
    fn enough(&self) -> u8 {
        // Until both kinds have been timed, the call counts as a short one.
        if self.compiled_best == u64::MAX || self.interpreted_best == u64::MAX {
            return TRIAL_SAMPLES;
        }
        let faster = self.compiled_best.min(self.interpreted_best);
        let fit = trial_calls(faster, TRIAL_SLOW_SAMPLES as u32, TRIAL_SAMPLES as u32) as u8;
        let behind_by = |shift: u32| {
            self.settled_best != u64::MAX
                && self.interpreted_best + (self.interpreted_best >> shift) < self.settled_best
        };
        if behind_by(TRIAL_MARGIN_SHIFT) && !behind_by(TRIAL_CLEAR_SHIFT) {
            fit.max(TRIAL_CLOSE_SAMPLES)
        } else {
            fit
        }
    }

    fn start(epoch: u32) -> Trial {
        Trial {
            phase: TRIAL_WARMUP,
            epoch,
            born: ENTRY_CALLS.load(Ordering::Relaxed) as u32,
            calls: 0,
            compiled_best: u64::MAX,
            compiled_samples: 0,
            settled_best: u64::MAX,
            settled_samples: 0,
            interpreted_best: u64::MAX,
            interpreted_samples: 0,
        }
    }
}

const TRIAL_WARMUP: u8 = 0;
const TRIAL_SAMPLING: u8 = 1;
const TRIAL_COMPILED: u8 = 2;
const TRIAL_DEMOTED: u8 = 3;
const TRIAL_WARMUP_CALLS: u32 = 8;
const TRIAL_SAMPLES: u8 = 10;
/// With this many samples of each, one side that is a quarter faster than
/// the other has won: the rest of the samples would be calls at the speed of
/// the slower one for nothing.
const TRIAL_EARLY_SAMPLES: u8 = 3;
/// A long call is measured with fewer calls: every call of the slower kind
/// is long, and a time that long varies less from one call to the next. As
/// many calls of each kind as fit into this much time are taken, and at
/// least `TRIAL_SLOW_SAMPLES`; as many calls as fit warm up, and at least
/// `TRIAL_SLOW_WARMUP_CALLS`.
const TRIAL_BUDGET_NS: u64 = 2_500_000_000;
const TRIAL_SLOW_WARMUP_CALLS: u32 = 3;
const TRIAL_SLOW_SAMPLES: u8 = 2;
/// Compiled code that is behind by less than 1/8 is not given up on fewer
/// samples than this: a couple of calls differ by that much on their own.
const TRIAL_CLOSE_SAMPLES: u8 = 5;
const TRIAL_CLEAR_SHIFT: u32 = 3;

/// How many of `most` calls to spend on calls that take `ns`.
fn trial_calls(ns: u64, least: u32, most: u32) -> u32 {
    ((TRIAL_BUDGET_NS / ns.max(1)).min(most as u64) as u32).max(least)
}
/// A measurement that never gets quiet enough is over after this many calls.
const TRIAL_MAX_CALLS: u32 = 400;
/// How many calls there have been of the functions that were given to be
/// compiled (not of what they call).
static ENTRY_CALLS: AtomicU64 = AtomicU64::new(0);
/// A measurement that is over within this many of them started while what
/// is below it was still being compiled, and starts again: a callee of a
/// loop is called `TRIAL_MAX_CALLS` times before the first call of the
/// function it is in returns.
const TRIAL_YOUNG_ENTRIES: u32 = 16;
/// How much faster the interpreter has to be: 1/32, about 3%.
const TRIAL_MARGIN_SHIFT: u32 = 5;

/// Bumped when what compiled code does changes for a reason the VM does not
/// see, such as a pass replacing part of a block by a compiled graph. A
/// measurement that is under way starts over: it would compare the
/// interpreter with code that is no longer what runs.
static PLAN_EPOCH: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// Counts the calls that are part of a measurement: a sample during which
/// it did not move had nothing below it that was still being measured.
static TRIAL_EVENTS: AtomicU64 = AtomicU64::new(0);
/// How many of them are under way, one inside the other, and the time the
/// outermost ones have taken.
static TRIAL_DEPTH: AtomicU64 = AtomicU64::new(0);
static TRIAL_TIME: AtomicU64 = AtomicU64::new(0);
/// Measurements below a function are worth mentioning when the calls they
/// are made of take more than 1/64 of a call of it: what they decide cannot
/// change its time by much more than that.
const TRIAL_NOTICEABLE_SHIFT: u32 = 6;

pub fn plan_changed() {
    PLAN_EPOCH.fetch_add(1, Ordering::Relaxed);
}

/// Counts as a call that is part of a measurement: the samples of the
/// functions that are running do not count as settled.
pub fn unsettled() {
    TRIAL_EVENTS.fetch_add(1, Ordering::Relaxed);
}

/// What waits for a device to be done with what it was given, once a
/// compiled graph runs on one: a builtin, kept for good. A kernel on a GPU
/// is started and not waited for, so the time of a call says little: a
/// graph that waits somewhere inside would look slower than the same
/// kernels started one by one, and the function would be handed to the
/// interpreter for it.
static TRIAL_SETTLE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// A sample that took this long is one that may have started such work.
const TRIAL_SETTLE_NS: u64 = 100_000;

pub fn trial_settles(settle: Obj) {
    if !settle.is_null() {
        unsafe { ffi::Py_INCREF(settle) };
    }
    let before = TRIAL_SETTLE.swap(settle as usize, Ordering::Relaxed) as Obj;
    if !before.is_null() {
        unsafe { ffi::Py_DECREF(before) };
    }
}

/// Waits for the devices, if anything runs on one. Whatever is raised is
/// dropped: a sample is then of the time until the call returned.
unsafe fn trial_settle() {
    let settle = TRIAL_SETTLE.load(Ordering::Relaxed) as Obj;
    if settle.is_null() {
        return;
    }
    let raised = ffi::PyErr_GetRaisedException();
    let done = ffi::PyObject_CallNoArgs(settle);
    if done.is_null() {
        ffi::PyErr_Clear();
    } else {
        ffi::Py_DECREF(done);
    }
    if !raised.is_null() {
        ffi::PyErr_SetRaisedException(raised);
    }
}

/// The builtins `locals`, `vars` and `dir`, which look for the variables of
/// the frame that calls them (see `Op::Locals`). Kept for good.
unsafe fn frame_readers() -> (Obj, Obj, Obj) {
    static FOUND: std::sync::OnceLock<(usize, usize, usize)> = std::sync::OnceLock::new();
    let found = FOUND.get_or_init(|| {
        let builtins = ffi::PyImport_ImportModule(c"builtins".as_ptr());
        let get = |name: &std::ffi::CStr| {
            let found = if builtins.is_null() {
                null_mut()
            } else {
                ffi::PyObject_GetAttrString(builtins, name.as_ptr())
            };
            if found.is_null() {
                ffi::PyErr_Clear();
            }
            found as usize
        };
        let all = (get(c"locals"), get(c"vars"), get(c"dir"));
        xdecref(builtins);
        all
    });
    (found.0 as Obj, found.1 as Obj, found.2 as Obj)
}

static TRIAL_DRAW: AtomicU64 = AtomicU64::new(0x9E37_79B9_7F4A_7C15);

unsafe impl Send for SettledBlocks {}
unsafe impl Sync for SettledBlocks {}

/// Per-name caches; only touched with the GIL held.
struct LoadCaches(Vec<std::cell::Cell<LoadCache>>);

unsafe impl Send for LoadCaches {}
unsafe impl Sync for LoadCaches {}

/// Lowers a Python function to the graph the VM runs.
pub fn lower(
    py: pyo3::Python<'_>,
    table: &OpcodeTable,
    func: &pyo3::Bound<'_, pyo3::PyAny>,
    options: &Options,
) -> pyo3::PyResult<Function> {
    use pyo3::prelude::*;
    let mut function = frontend::build(py, table, func)?;
    if options.inline {
        inline::inline_calls(py, table, &mut function, &func.getattr("__code__")?);
    }
    if options.unroll {
        function.unrolled = unroll::unroll_loops(&mut function, options.max_trip_count);
    }
    Ok(function)
}

/// The code of a function that compiled code called, compiled.
struct Callee {
    code: PyObj,
    _globals: PyObj,
    compiled: Arc<Compiled>,
}

/// The Python functions called from a function compiled with
/// torch.vm, or from anything compiled under it, each compiled the
/// first time it is called.
pub struct Callees {
    table: OpcodeTable,
    /// Keyed by the addresses of the code object and of the globals, which
    /// the entry keeps alive. Function objects are not kept: every function
    /// made from the same code shares the entry, and its closure and
    /// defaults are read from it at each call.
    entries: Shared<HashMap<(usize, usize), Callee>>,
    /// What a call finds without hashing: per slot, the code object and
    /// globals of the last function looked up there, and its compiled form
    /// (INTERPRETED if the frontend turned it down). The entries own what
    /// the pointers refer to.
    recent: Vec<std::cell::Cell<(usize, usize, *const Compiled)>>,
    /// Code objects the frontend does not support, by address: their
    /// functions run in the interpreter.
    interpreted: Shared<HashMap<usize, PyObj>>,
    /// Code objects that are left to the interpreter for where they are
    /// from (see `leave_to_interpreter`), by address.
    left: Shared<HashMap<usize, PyObj>>,
    call_name: PyObj,
    /// Set for a call that runs in the profiling tier, until the callee of
    /// that call has been looked at; and the Python function it came down
    /// to, with how.
    callee_asked: std::cell::Cell<bool>,
    callee_seen: std::cell::RefCell<Option<(PyObj, CalleeKind)>>,
}

unsafe impl Send for Callees {}
unsafe impl Sync for Callees {}

const MAX_CALLEES: usize = 4096;
const RECENT_CALLEES: usize = 256;
/// Stands for "runs in the interpreter" in the recent-callee slots: the
/// frontend turned the function down.
const INTERPRETED: *const Compiled = 1 as *const Compiled;
/// The same for a function that is left to the interpreter with all it
/// calls: it was measured to run faster there, or is one of the compilers'.
const DEMOTED: *const Compiled = 2 as *const Compiled;

impl Callees {
    pub fn new(table: OpcodeTable) -> Callees {
        Callees {
            table,
            entries: Shared::new(HashMap::new()),
            recent: (0..RECENT_CALLEES).map(|_| std::cell::Cell::new((0, 0, std::ptr::null()))).collect(),
            interpreted: Shared::new(HashMap::new()),
            left: Shared::new(HashMap::new()),
            call_name: unsafe {
                PyObj::from_owned(ffi::PyUnicode_InternFromString(c"__call__".as_ptr()))
            },
            callee_asked: std::cell::Cell::new(false),
            callee_seen: std::cell::RefCell::new(None),
        }
    }

    /// Qualified names of the functions that were compiled, and of those the
    /// frontend turned down.
    pub fn summary(&self) -> (Vec<String>, Vec<String>) {
        let name = |obj: Obj, attr: &std::ffi::CStr| unsafe {
            let name = ffi::PyObject_GetAttrString(obj, attr.as_ptr());
            if name.is_null() {
                ffi::PyErr_Clear();
                return "?".to_string();
            }
            let text = pyffi::str_to_string(name);
            ffi::Py_DECREF(name);
            text
        };
        let mut compiled: Vec<String> =
            self.entries.borrow().values().map(|c| name(c.code.as_ptr(), c"co_qualname")).collect();
        let mut interpreted: Vec<String> =
            self.interpreted.borrow().values().map(|c| name(c.as_ptr(), c"co_qualname")).collect();
        compiled.sort();
        interpreted.sort();
        (compiled, interpreted)
    }

    /// The compiled forms of the functions that were called, in the order
    /// of their files, lines and names.
    pub fn compiled(&self) -> Vec<Arc<Compiled>> {
        let mut all: Vec<Arc<Compiled>> = self.entries.borrow().values().map(|callee| callee.compiled.clone()).collect();
        all.sort_by(|a, b| {
            let key = |c: &Compiled| (c.func.filename.clone(), c.func.firstlineno, c.func.qualname.clone());
            key(a).cmp(&key(b))
        });
        all
    }

    /// Qualified names of the compiled functions that were measured to run
    /// faster in the interpreter and were left to it.
    pub fn demoted(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .entries
            .borrow()
            .values()
            .filter(|c| c.compiled.settled.trial.get().phase == TRIAL_DEMOTED)
            .map(|c| unsafe {
                let name = ffi::PyObject_GetAttrString(c.code.as_ptr(), c"co_qualname".as_ptr());
                if name.is_null() {
                    ffi::PyErr_Clear();
                    return "?".to_string();
                }
                let text = pyffi::str_to_string(name);
                ffi::Py_DECREF(name);
                text
            })
            .collect();
        names.sort();
        names
    }

    /// Drops everything compiled so far, which also breaks the reference
    /// cycle between this table and what it compiled.
    pub fn clear(&self) {
        for slot in &self.recent {
            slot.set((0, 0, std::ptr::null()));
        }
        let dropped = std::mem::take(&mut *self.entries.borrow_mut());
        drop(dropped);
        self.interpreted.borrow_mut().clear();
        self.left.borrow_mut().clear();
    }
}

/// Where the code is that is not compiled: a file, a directory (with a
/// trailing separator), or one function of a file (`file::qualname`). It is
/// what compiles and what was compiled: the graph compilers and the wrappers
/// they put around their graphs, which are reached from a compiled function
/// when it calls a function that Dynamo compiled, or when autograd runs the
/// backward of a compiled graph.
static LEFT: std::sync::RwLock<Vec<String>> = std::sync::RwLock::new(Vec::new());

/// Leaves the functions whose code is in one of `places` to the
/// interpreter, with all they call.
pub fn leave_to_interpreter(places: Vec<String>) {
    let mut left = LEFT.write().unwrap();
    for place in places {
        if !left.contains(&place) {
            left.push(place);
        }
    }
}

unsafe fn is_left(code: Obj) -> bool {
    let left = LEFT.read().unwrap();
    if left.is_empty() {
        return false;
    }
    let filename = ffi::PyObject_GetAttrString(code, c"co_filename".as_ptr());
    if filename.is_null() {
        ffi::PyErr_Clear();
        return false;
    }
    let name = pyffi::str_to_string(filename);
    ffi::Py_DECREF(filename);
    left.iter().any(|place| match place.split_once("::") {
        Some((file, function)) => {
            name == file && {
                let qualname = ffi::PyObject_GetAttrString(code, c"co_qualname".as_ptr());
                if qualname.is_null() {
                    ffi::PyErr_Clear();
                    return false;
                }
                let found = pyffi::str_to_string(qualname);
                ffi::Py_DECREF(qualname);
                found == function
            }
        }
        None if place.ends_with('/') => name.starts_with(place.as_str()),
        None => name == *place,
    })
}

/// The positional arguments of a call, wherever the caller has them: the
/// object a method is bound to, if there is one, and then `n` values.
#[derive(Clone, Copy)]
struct Args {
    receiver: Obj,
    items: *const Obj,
    n: usize,
}

impl Args {
    unsafe fn tuple(receiver: Obj, tuple: Obj) -> Args {
        let items = std::ptr::addr_of!((*(tuple as *mut ffi::PyTupleObject)).ob_item) as *const Obj;
        Args { receiver, items, n: ffi::PyTuple_GET_SIZE(tuple) as usize }
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.n + !self.receiver.is_null() as usize
    }

    #[inline(always)]
    unsafe fn get(&self, i: usize) -> Obj {
        if self.receiver.is_null() {
            *self.items.add(i)
        } else if i == 0 {
            self.receiver
        } else {
            *self.items.add(i - 1)
        }
    }

    /// A new tuple of the arguments from index `from` on.
    unsafe fn slice(&self, from: usize) -> Obj {
        let n = self.len();
        let tuple = ffi::PyTuple_New(n.saturating_sub(from) as isize);
        if !tuple.is_null() {
            for i in from..n {
                ffi::PyTuple_SET_ITEM(tuple, (i - from) as isize, new_ref(self.get(i)));
            }
        }
        tuple
    }
}

/// The interpreter frame a compiled function runs under. The VM keeps its
/// instruction pointer on the bytecode of the instruction it executes, and
/// its variables what the function binds them to (see `Op::StoreLocal`); the
/// values the function computes with live in the VM's registers.
#[derive(Clone, Copy)]
struct Frame {
    /// Null where frames are not supported.
    ptr: *mut std::ffi::c_void,
    instr: *mut *mut u16,
    code: *mut u16,
    /// The variables of the frame: an owned reference or null each, the
    /// way the interpreter keeps one (see `frame_ref`).
    locals: *mut Obj,
    /// Where the frame has its frame object: null until something looks at
    /// the frame (a traceback, a debugger).
    object: *mut Obj,
}

/// What a variable of an interpreter frame holds for an owned reference, or
/// for null: in 3.12 just that.
#[cfg(not(all(Py_3_14, not(Py_GIL_DISABLED), target_pointer_width = "64")))]
#[inline(always)]
unsafe fn frame_ref(owned: Obj) -> Obj {
    owned
}

/// What is to be released of what a variable of a frame held.
#[cfg(not(all(Py_3_14, not(Py_GIL_DISABLED), target_pointer_width = "64")))]
#[inline(always)]
unsafe fn frame_owned(held: Obj) -> Obj {
    held
}

/// The object a variable of a frame holds, borrowed, or null.
#[cfg(not(all(Py_3_14, not(Py_GIL_DISABLED), target_pointer_width = "64")))]
#[inline(always)]
unsafe fn frame_value(held: Obj) -> Obj {
    held
}

/// What a variable of an interpreter frame holds for an owned reference, or
/// for null. In 3.14 it is a stack reference: the pointer, with its lowest
/// bit set if the reference is not the variable's to release. That bit alone
/// is NULL, and a reference to an immortal object always has it, as
/// PyStackRef_FromPyObjectSteal makes them in the default build, the only
/// one frames are pushed in (see cfg.cpp).
#[cfg(all(Py_3_14, not(Py_GIL_DISABLED), target_pointer_width = "64"))]
#[inline(always)]
unsafe fn frame_ref(owned: Obj) -> Obj {
    if owned.is_null() {
        return 1 as Obj;
    }
    (owned as usize | ((*owned).ob_refcnt.refcnt_and_flags.ob_flags & 1) as usize) as Obj
}

/// What is to be released of what a variable of a frame held.
#[cfg(all(Py_3_14, not(Py_GIL_DISABLED), target_pointer_width = "64"))]
#[inline(always)]
unsafe fn frame_owned(held: Obj) -> Obj {
    if held as usize & 1 == 0 {
        held
    } else {
        null_mut()
    }
}

/// The object a variable of a frame holds, borrowed, or null: what the
/// interpreter bound an argument to, in a frame it made.
#[cfg(all(Py_3_14, not(Py_GIL_DISABLED), target_pointer_width = "64"))]
#[inline(always)]
unsafe fn frame_value(held: Obj) -> Obj {
    (held as usize & !1) as Obj
}

impl Frame {
    const NONE: Frame =
        Frame { ptr: null_mut(), instr: null_mut(), code: null_mut(), locals: null_mut(), object: null_mut() };

    unsafe fn push(function: Obj) -> Result<Frame, ()> {
        let mut frame = Frame::NONE;
        if let Some(push) = capi::api().and_then(|api| api.frame_push) {
            frame.ptr = push(function, &mut frame.instr, &mut frame.code, &mut frame.locals, &mut frame.object);
            if frame.ptr.is_null() {
                return Err(());
            }
        }
        Ok(frame)
    }

    /// A frame object that outlives the frame takes its variables with it.
    unsafe fn pop(self) {
        if !self.ptr.is_null() {
            (capi::api().unwrap_unchecked().frame_pop.unwrap_unchecked())(self.ptr);
        }
    }

    /// Whether something refers to the frame object: the traceback of the
    /// exception that is leaving the function, as a rule.
    unsafe fn held(&self) -> bool {
        !self.ptr.is_null()
            && capi::api().and_then(|api| api.frame_held).is_some_and(|held| held(self.ptr) != 0)
    }

    /// The instruction the frame is on, as the index the interpreter calls
    /// lasti (a new reference to an int); None without a frame.
    unsafe fn lasti(&self) -> Obj {
        if self.instr.is_null() {
            return new_ref(ffi::Py_None());
        }
        ffi::PyLong_FromSsize_t((*self.instr).offset_from(self.code))
    }

    /// Puts the frame back on the instruction `lasti` named.
    unsafe fn restore(&self, lasti: Obj) {
        if !self.instr.is_null() && !lasti.is_null() && ffi::PyLong_Check(lasti) != 0 {
            *self.instr = self.code.offset(ffi::PyLong_AsSsize_t(lasti));
        }
    }

    /// Records that the instruction at this offset of the function's
    /// bytecode is the one being executed.
    #[inline(always)]
    unsafe fn at(&self, offset: u32) {
        if !self.instr.is_null() {
            *self.instr = self.code.add(offset as usize / 2);
        }
    }
}

const OFFSET_MASK: u32 = (1 << ORIGIN_SHIFT) - 1;

/// The frame of a compiled function and, above it, those of the inlined
/// functions that are running in it.
struct Frames {
    frames: [Frame; Frames::CAPACITY],
    /// How many inlined functions deep the VM is.
    depth: usize,
}

impl Frames {
    /// Inlining nests four deep at most.
    const CAPACITY: usize = 6;

    fn new(root: Frame) -> Frames {
        Frames { frames: [root; Frames::CAPACITY], depth: 0 }
    }

    #[inline(always)]
    fn top(&self) -> Frame {
        if self.depth < Frames::CAPACITY {
            self.frames[self.depth]
        } else {
            Frame::NONE
        }
    }

    /// The depth grows even when the frame could not be pushed, so that
    /// every enter has its leave.
    unsafe fn enter(&mut self, function: Obj) -> Result<(), ()> {
        self.depth += 1;
        if self.depth < Frames::CAPACITY {
            self.frames[self.depth] = Frame::NONE;
            self.frames[self.depth] = Frame::push(function)?;
        }
        Ok(())
    }

    unsafe fn leave(&mut self) {
        if self.depth == 0 {
            return;
        }
        if self.depth < Frames::CAPACITY {
            self.frames[self.depth].pop();
        }
        self.depth -= 1;
    }
}

/// The variables of frames that are bound to a tensor without a Python
/// object (see `Compiled::obj`): the frame, the variable's place in it, which
/// is empty meanwhile, and a reference to the tensor. Each gets its object,
/// and the frame the object, before anything runs that can look at a frame:
/// a call, an exception, a signal check. So there are none whenever the
/// function is not the one that is running.
struct Unbound {
    list: std::cell::RefCell<Vec<(*mut std::ffi::c_void, *mut Obj, Obj)>>,
    any: std::cell::Cell<bool>,
}

unsafe impl Send for Unbound {}
unsafe impl Sync for Unbound {}

/// What the interpreter says of an object that `with` cannot use, for lack
/// of the method `name`.
#[cfg(not(Py_3_14))]
unsafe fn not_a_context_manager(o: Obj, name: &str) -> String {
    let missed = if name == "__exit__" { " (missed __exit__ method)" } else { "" };
    format!("'{}' object does not support the context manager protocol{}", pyffi::type_name(o), missed)
}

/// What the interpreter says of an object that `with` cannot use, for lack
/// of the method `name`. From 3.14 on it names the type in full and either
/// method, and points an asynchronous context manager to `async with`.
#[cfg(Py_3_14)]
unsafe fn not_a_context_manager(o: Obj, name: &str) -> String {
    let ty = ffi::Py_TYPE(o);
    let full = ffi::PyType_GetFullyQualifiedName(ty);
    let type_name = if full.is_null() {
        ffi::PyErr_Clear();
        pyffi::type_name(o)
    } else {
        let text = pyffi::str_to_string(full);
        ffi::Py_DECREF(full);
        text
    };
    let asynchronous = [c"__aenter__", c"__aexit__"].iter().all(|method| {
        let method = ffi::PyUnicode_InternFromString(method.as_ptr());
        let found = if method.is_null() { null_mut() } else { pyffi::_PyType_Lookup(ty, method) };
        xdecref(method);
        !found.is_null() && (*ffi::Py_TYPE(found)).tp_descr_get.is_some()
    });
    ffi::PyErr_Clear();
    let hint = if asynchronous {
        " but it supports the asynchronous context manager protocol. Did you mean to use 'async with'?"
    } else {
        ""
    };
    format!("'{}' object does not support the context manager protocol (missed {} method){}", type_name, name, hint)
}

/// `match_keys` of ceval.c: the values `map` has for `keys`, as a tuple; None
/// if it lacks one of them; null with an error set if looking raised.
unsafe fn match_keys(map: Obj, keys: Obj) -> Obj {
    let nkeys = ffi::PyTuple_GET_SIZE(keys);
    if nkeys == 0 {
        return ffi::PyTuple_New(0);
    }
    // The two-argument form of get: it neither raises for a missing key nor
    // creates one in a mapping that defines __missing__.
    let name = ffi::PyUnicode_InternFromString(c"get".as_ptr());
    let mut get: Obj = null_mut();
    let unbound = if name.is_null() { 0 } else { pyffi::_PyObject_GetMethod(map, name, &mut get) };
    xdecref(name);
    if get.is_null() {
        return null_mut();
    }
    let seen = ffi::PySet_New(null_mut());
    let dummy = ffi::PyObject_CallNoArgs(std::ptr::addr_of_mut!(ffi::PyBaseObject_Type) as Obj);
    let mut values = ffi::PyTuple_New(nkeys);
    let mut ok = !seen.is_null() && !dummy.is_null() && !values.is_null();
    let mut i = 0;
    while ok && i < nkeys {
        let key = ffi::PyTuple_GET_ITEM(keys, i);
        if ffi::PySet_Contains(seen, key) != 0 || ffi::PySet_Add(seen, key) != 0 {
            if ffi::PyErr_Occurred().is_null() {
                set_error(
                    ffi::PyExc_ValueError,
                    format!("mapping pattern checks duplicate key ({})", pyffi::repr(key)),
                );
            }
            ok = false;
            break;
        }
        let args = [map, key, dummy];
        let value = if unbound == 1 {
            ffi::PyObject_Vectorcall(get, args.as_ptr(), 3, null_mut())
        } else {
            ffi::PyObject_Vectorcall(get, args.as_ptr().add(1), 2, null_mut())
        };
        if value.is_null() {
            ok = false;
        } else if value == dummy {
            // The key is not in the mapping: no match.
            ffi::Py_DECREF(value);
            ffi::Py_DECREF(values);
            values = new_ref(ffi::Py_None());
            break;
        } else {
            ffi::PyTuple_SET_ITEM(values, i, value);
        }
        i += 1;
    }
    xdecref(get);
    xdecref(seen);
    xdecref(dummy);
    if !ok {
        xdecref(values);
        return null_mut();
    }
    values
}

/// `match_class_attr` of ceval.c: the attribute, or null, with an error set
/// only if the name was looked up before or the lookup raised something
/// other than AttributeError.
unsafe fn match_class_attr(subject: Obj, class: Obj, name: Obj, seen: Obj) -> Obj {
    if ffi::PySet_Contains(seen, name) != 0 || ffi::PySet_Add(seen, name) != 0 {
        if ffi::PyErr_Occurred().is_null() {
            set_error(
                ffi::PyExc_TypeError,
                format!(
                    "{}() got multiple sub-patterns for attribute {}",
                    pyffi::class_name(class),
                    pyffi::repr(name)
                ),
            );
        }
        return null_mut();
    }
    let attr = ffi::PyObject_GetAttr(subject, name);
    if attr.is_null() && ffi::PyErr_ExceptionMatches(ffi::PyExc_AttributeError) != 0 {
        ffi::PyErr_Clear();
    }
    attr
}

/// `match_class` of ceval.c: the attributes a class pattern with `nargs`
/// positional sub-patterns and the keyword sub-patterns `names` binds, as a
/// tuple; None if the subject does not match; null with an error set.
unsafe fn match_class(subject: Obj, class: Obj, nargs: isize, names: Obj) -> Obj {
    const MATCH_SELF: std::os::raw::c_ulong = 1 << 22;
    if ffi::PyType_Check(class) == 0 {
        set_error(ffi::PyExc_TypeError, "called match pattern must be a class".to_string());
        return null_mut();
    }
    let no_match = || -> Obj {
        if ffi::PyErr_Occurred().is_null() {
            new_ref(ffi::Py_None())
        } else {
            null_mut()
        }
    };
    if ffi::PyObject_IsInstance(subject, class) <= 0 {
        return no_match();
    }
    let seen = ffi::PySet_New(null_mut());
    let attrs = ffi::PyList_New(0);
    let mut ok = !seen.is_null() && !attrs.is_null();
    if ok && nargs > 0 {
        let mut match_self = false;
        let mut match_args = ffi::PyObject_GetAttrString(class, c"__match_args__".as_ptr());
        if match_args.is_null() {
            if ffi::PyErr_ExceptionMatches(ffi::PyExc_AttributeError) != 0 {
                ffi::PyErr_Clear();
                // Only a type without __match_args__ matches as itself.
                match_args = ffi::PyTuple_New(0);
                match_self =
                    ffi::PyType_HasFeature(class as *mut ffi::PyTypeObject, MATCH_SELF) != 0;
            }
            ok = !match_args.is_null();
        } else if ffi::PyTuple_CheckExact(match_args) == 0 {
            set_error(
                ffi::PyExc_TypeError,
                format!(
                    "{}.__match_args__ must be a tuple (got {})",
                    pyffi::class_name(class),
                    pyffi::type_name(match_args)
                ),
            );
            ok = false;
        }
        if ok {
            let allowed = if match_self { 1 } else { ffi::PyTuple_GET_SIZE(match_args) };
            if allowed < nargs {
                set_error(
                    ffi::PyExc_TypeError,
                    format!(
                        "{}() accepts {} positional sub-pattern{} ({} given)",
                        pyffi::class_name(class),
                        allowed,
                        if allowed == 1 { "" } else { "s" },
                        nargs
                    ),
                );
                ok = false;
            } else if match_self {
                ok = ffi::PyList_Append(attrs, subject) == 0;
            } else {
                for i in 0..nargs {
                    let name = ffi::PyTuple_GET_ITEM(match_args, i);
                    if ffi::PyUnicode_CheckExact(name) == 0 {
                        set_error(
                            ffi::PyExc_TypeError,
                            format!(
                                "__match_args__ elements must be strings (got {})",
                                pyffi::type_name(name)
                            ),
                        );
                        ok = false;
                        break;
                    }
                    let attr = match_class_attr(subject, class, name, seen);
                    ok = !attr.is_null() && ffi::PyList_Append(attrs, attr) == 0;
                    xdecref(attr);
                    if !ok {
                        break;
                    }
                }
            }
        }
        xdecref(match_args);
    }
    if ok {
        for i in 0..ffi::PyTuple_GET_SIZE(names) {
            let attr = match_class_attr(subject, class, ffi::PyTuple_GET_ITEM(names, i), seen);
            ok = !attr.is_null() && ffi::PyList_Append(attrs, attr) == 0;
            xdecref(attr);
            if !ok {
                break;
            }
        }
    }
    let result = if ok { ffi::PyList_AsTuple(attrs) } else { no_match() };
    xdecref(seen);
    xdecref(attrs);
    result
}

/// Does what the interpreter does when it checks its eval breaker: run
/// signal handlers and pending calls, let a scheduled garbage collection
/// happen, hand the GIL to a thread that asked for it, raise an asynchronous
/// exception. There is no entry to that from outside the interpreter, so
/// this calls a Python function that does nothing: the interpreter checks on
/// entering it. A loop of compiled code calls this every 64 iterations.
/// Returns false with an error set if something was raised.
unsafe fn tick() -> bool {
    static NOTHING: std::sync::atomic::AtomicPtr<ffi::PyObject> =
        std::sync::atomic::AtomicPtr::new(null_mut());
    let mut nothing = NOTHING.load(Ordering::Relaxed);
    if nothing.is_null() {
        let py = pyo3::Python::assume_attached();
        match py.eval(c"lambda: None", None, None) {
            Ok(function) => {
                nothing = function.into_ptr();
                NOTHING.store(nothing, Ordering::Relaxed);
            }
            Err(err) => {
                err.restore(py);
                return false;
            }
        }
    }
    // The interpreter is to run it: the VM would not look at what is pending.
    let _nobody = Hooked::nobody();
    let result = ffi::PyObject_CallNoArgs(nothing);
    if result.is_null() {
        return false;
    }
    ffi::Py_DECREF(result);
    true
}

thread_local! {
    /// Above zero while a range of a block is being replayed for a trace:
    /// the functions it calls are then left to the interpreter, whose calls
    /// the tracer sees.
    static REPLAY_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// The frame of the function that is calling a region (`Op::CallObject`):
    /// what the instructions of the region run under when the
    /// region replays them (see `Compiled::replay`).
    static REGION_FRAME: std::cell::Cell<Frame> = const { std::cell::Cell::new(Frame::NONE) };
    /// How many blocks are running in the profiling tier on this thread, one
    /// inside the other.
    static PROFILING_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// The function handed to torch.vm whose call is running on this
    /// thread: the frames the interpreter starts meanwhile are offered to it
    /// (see `take_frame`). Null while what runs is not the function's own
    /// code but the runtime's: a pass, a region, an import, a call that is
    /// measured in the interpreter.
    static HOOKED: std::cell::Cell<*const Compiled> = const { std::cell::Cell::new(std::ptr::null()) };
}

/// What `exec_profiled` takes down in front of an instruction that can run
/// code that is not the VM's, for the facts it finds behind it.
struct Probe {
    /// How many frames the interpreter has counted.
    frames: u64,
    /// The tensors the instruction is given, each with its version counter.
    versions: Vec<(Obj, u64)>,
    /// For a call: what a call further up was waiting to learn of its callee.
    outer: Option<(bool, Option<(PyObj, CalleeKind)>)>,
}

impl Probe {
    unsafe fn start(callees: &Callees, regs: &[Obj], op: &Op, api: &TcfgApi) -> Probe {
        // The interpreter counts the frames it starts while they are wanted.
        if let (Some(want), Some(_)) = (api.frame_want, api.frame_count) {
            want(1);
        }
        let mut given: Vec<Obj> = Vec::new();
        op.for_each_use(|v| given.push(reg(regs, v)));
        if let Op::Call { callee, .. } = op {
            // The receiver of a method that was looked up earlier.
            let c = reg(regs, *callee);
            if !c.is_null() && !capi::is_unboxed(c) && ffi::PyCFunction_Check(c) != 0 {
                given.push(ffi::PyCFunction_GetSelf(c));
            }
        }
        let versions = versions_of(&given, api);
        let call = matches!(op, Op::Call { .. } | Op::CallMethod { .. } | Op::CallEx { .. });
        let outer = call.then(|| (callees.callee_asked.replace(true), callees.callee_seen.take()));
        Probe { frames: api.frame_count.map_or(0, |count| count()), versions, outer }
    }

    unsafe fn finish(self, callees: &Callees, facts: &mut InstFacts, api: &TcfgApi) {
        // Where the interpreter does not tell of its frames, one may have
        // been entered.
        facts.frames |= api.frame_count.is_none_or(|count| count() != self.frames);
        if let (Some(want), Some(_)) = (api.frame_want, api.frame_count) {
            want(-1);
        }
        facts.wrote |= versions_moved(&self.versions, api);
        if let Some((asked, seen)) = self.outer {
            callees.callee_asked.set(asked);
            // A function the VM runs itself has no frame of the
            // interpreter's making to count.
            if let Some((code, kind)) = callees.callee_seen.replace(seen) {
                facts.frames = true;
                facts.saw_callee(code, kind);
            }
        }
    }
}

/// The tensors among some values that count the writes to them, each with
/// its version counter.
unsafe fn versions_of(values: &[Obj], api: &TcfgApi) -> Vec<(Obj, u64)> {
    let mut versions = Vec::new();
    for o in values {
        let mut info = TcfgTensorInfo::empty();
        if !o.is_null() && (api.tensor_info)(*o, &mut info) != 0 && info.version != capi::TCFG_NO_VERSION {
            versions.push((*o, info.version));
        }
    }
    versions
}

/// Whether one of the tensors was written to in place since.
unsafe fn versions_moved(versions: &[(Obj, u64)], api: &TcfgApi) -> bool {
    versions.iter().any(|(tensor, version)| {
        let mut info = TcfgTensorInfo::empty();
        (api.tensor_info)(*tensor, &mut info) != 0 && info.version != *version
    })
}

/// Some values and, of the lists and tuples among them, what they hold.
unsafe fn flattened(values: &[Obj]) -> Vec<Obj> {
    let mut all = Vec::new();
    for o in values.iter().filter(|o| !o.is_null()) {
        all.push(*o);
        if capi::is_unboxed(*o) {
            continue;
        }
        if ffi::PyList_CheckExact(*o) != 0 {
            all.extend((0..ffi::PyList_GET_SIZE(*o)).map(|i| ffi::PyList_GET_ITEM(*o, i)));
        } else if ffi::PyTuple_CheckExact(*o) != 0 {
            all.extend((0..ffi::PyTuple_GET_SIZE(*o)).map(|i| ffi::PyTuple_GET_ITEM(*o, i)));
        }
    }
    all
}

/// How entering a graph ended (see `Compiled::enter_graph`).
enum PathEnd {
    /// Nothing was done: the instructions run.
    Miss,
    /// The work up to an instruction is done: the function goes on there.
    Landed(Landing),
    /// An instruction of a block raised: the block, and the offset of the
    /// instruction's bytecode.
    Raised(usize, u32),
    /// A guard of a block failed while its instructions were taken down:
    /// the block, and the generic instruction to resume at.
    Deopt(usize, usize),
}

/// Where a function goes on: at an instruction of some code of a block (the
/// generic code, for null), or at the block itself, entered with what an
/// edge to it does already done (`AFRESH`).
#[derive(Clone, Copy)]
struct Landing {
    block: usize,
    code: *const BlockCode,
    pc: usize,
}

const AFRESH: usize = usize::MAX;

/// What became of one path that a call tried (see `Compiled::run_path`).
enum Ran {
    Landed(Landing),
    /// The path is not for this call: another may be.
    Other,
    /// The call is one that runs the instructions.
    Miss,
}

/// How a stage of a path ended (see `Compiled::run_stage`).
#[derive(PartialEq)]
enum Staged {
    Done,
    /// A check failed, or an instruction raised: the path is not for the
    /// call.
    Failed,
    /// An instruction could run Python code where it stands.
    Unsure,
}

/// What became of an instruction that is taken down (see
/// `Compiled::record_inst`).
enum Taken {
    Went,
    /// It is no part of a path, for this reason: the path ends in front of
    /// it.
    Ends(String),
    Raised,
    Deopt(usize),
    /// A call that the trace went into has run without the trace coming
    /// back from it, for this reason: what it returned (null: it raised).
    Cut(Obj, String),
}

/// What a path goes into in place of an instruction (see
/// `Compiled::follow`): a call; a call of a method; the function that the
/// class of an object has for what the interpreter does with the object,
/// the second value being the key of an item; the getter of a property,
/// the `__get__` of a descriptor, or the `__getattr__` of the class, for
/// an attribute of a name.
enum Form<'a> {
    Call { callee: V, args: &'a [V], kwnames: u32 },
    Method { recv: V, name: u32, args: &'a [V], kwnames: u32 },
    /// A call that hands on a sequence with `*` and a dict with `**`.
    Spread { callee: V, args: V, kwargs: V },
    Special(&'static PyObj, V, Option<V>),
    Property(V, u32),
    Descriptor(V, u32),
    Getattr(V, u32),
}

impl Form<'_> {
    fn text(&self) -> &'static str {
        match self {
            Form::Call { .. } | Form::Spread { .. } => "a call",
            Form::Method { .. } => "a call of a method",
            Form::Special(..) => "a function of a class",
            Form::Property(..) => "a property",
            Form::Descriptor(..) => "a descriptor",
            Form::Getattr(..) => "a __getattr__",
        }
    }
}

/// What became of a function that a path was to go into.
enum Followed {
    /// It was not gone into, and nothing has run.
    Not,
    /// It returned on the trace: what, and the value of the path that is.
    Returned(Obj, PathValue),
    Raised,
    /// It ran to its end without the trace coming back from it, for this
    /// reason: what it returned (null: it raised).
    Left(Obj, String),
}

/// What became of a function that is taken down (see `Compiled::walk`).
enum Walked {
    /// A called function returned: what, and the value of the path that is.
    Returned(Obj, PathValue),
    /// The path ends in front of instruction `pc` of `code`, or in front of
    /// the block, which was not profiled yet.
    Ended { block: usize, code: *const BlockCode, pc: usize, afresh: bool, reason: String },
    /// The trace did not come back from the call at instruction `call`,
    /// which has run: the function goes on behind it.
    Left { block: usize, code: *const BlockCode, call: usize, reason: String },
    Raised(usize, u32),
    Deopt(usize, usize),
}

/// How many instructions a path has at most.
const MAX_PATH_STEPS: usize = 200_000;

/// How many calls a path is inside of at most, and how many of them may be
/// calls of one function.
const MAX_PATH_DEPTH: usize = 48;
const MAX_PATH_AGAIN: usize = 8;

const NOT_PROFILED: &str = "a block that was not profiled yet";

thread_local! {
    /// How many calls on this thread are of a function whose first block is
    /// still in the profiling tier.
    static WARMING: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Counts a call of a function whose first block is still in the profiling
/// tier, for as long as the call lives: the graphs of what it calls wait
/// for the graph it may get itself, which would hold their operations.
struct Warming<'a>(Option<&'a Compiled>);

impl<'a> Warming<'a> {
    #[cold]
    fn begin(&mut self, compiled: &'a Compiled) {
        if self.0.is_none() {
            WARMING.with(|w| w.set(w.get() + 1));
            compiled.settled.warming.set(compiled.settled.warming.get() + 1);
            self.0 = Some(compiled);
        }
    }
}

impl Drop for Warming<'_> {
    fn drop(&mut self) {
        if let Some(compiled) = self.0 {
            WARMING.with(|w| w.set(w.get() - 1));
            compiled.settled.warming.set(compiled.settled.warming.get() - 1);
        }
    }
}

/// Says, for as long as it lives, whom the frames that the interpreter
/// starts on this thread are offered to.
struct Hooked {
    outer: *const Compiled,
}

impl Hooked {
    #[inline(always)]
    unsafe fn to(owner: *const Compiled) -> Hooked {
        let outer = HOOKED.with(|h| h.replace(owner));
        Hooked::want(outer, owner);
        Hooked { outer }
    }

    /// Nobody: the interpreter runs what it starts.
    #[inline(always)]
    unsafe fn nobody() -> Hooked {
        Hooked::to(std::ptr::null())
    }

    /// The interpreter calls the hook only while a thread wants it to, and
    /// makes its own calls faster when none does.
    #[inline(always)]
    unsafe fn want(from: *const Compiled, to: *const Compiled) {
        if from.is_null() != to.is_null() {
            if let Some(want) = capi::api().and_then(|api| api.frame_want) {
                want(if to.is_null() { -1 } else { 1 });
            }
        }
    }
}

impl Drop for Hooked {
    #[inline(always)]
    fn drop(&mut self) {
        unsafe {
            let inner = HOOKED.with(|h| h.replace(self.outer));
            Hooked::want(inner, self.outer);
        }
    }
}

/// Whether the frames the interpreter starts on this thread are offered to a
/// compiled function now.
pub fn frames_offered() -> bool {
    HOOKED.with(|h| !h.get().is_null())
}

/// What the interpreter's frame hook offers a frame to: one it made for a
/// call of the Python function `function`, with the arguments bound and
/// nothing run yet. These are the calls the VM does not make itself: those
/// of C code (a constructor, a property, an operator, a callback of
/// `sorted` or of autograd, `functools.partial`), and those of a function
/// that the interpreter runs under a compiled one. The function runs as a
/// callee of the function whose call this is: compiled, in the frame the
/// interpreter made; or by the interpreter, if the frontend turned it down
/// (what it calls is offered in turn) or it was measured to run faster
/// there (with all it calls). Returns 0 if no compiled function is running.
pub unsafe extern "C" fn take_frame(function: Obj, frame: *mut std::ffi::c_void, result: *mut Obj) -> i32 {
    let owner = HOOKED.with(|h| h.get());
    if owner.is_null() {
        return 0;
    }
    let owner = &*owner;
    let interpret = || (capi::api().unwrap_unchecked().frame_interpret.unwrap_unchecked())(frame);
    let form = owner.function_form(function);
    *result = if form == INTERPRETED {
        interpret()
    } else if form == DEMOTED {
        let _nobody = Hooked::nobody();
        interpret()
    } else {
        owner.tiered(form, function, || (*form).enter_frame(function, frame), interpret)
    };
    1
}

/// How many profiling runs of a block in a row may end in a raise before it
/// is specialized from what they saw.
const PROFILE_RAISES: u32 = 2;

/// How many blocks on this thread are in their profiling run: code that
/// decides when to compile something uses it to let the outermost block go
/// first.
pub fn profiling_depth() -> usize {
    PROFILING_DEPTH.with(|d| d.get())
}

/// Counts a block in the profiling tier for as long as it lives.
struct Profiling(bool);

impl Profiling {
    fn enter(active: bool) -> Profiling {
        if active {
            PROFILING_DEPTH.with(|d| d.set(d.get() + 1));
        }
        Profiling(active)
    }
}

impl Drop for Profiling {
    fn drop(&mut self) {
        if self.0 {
            PROFILING_DEPTH.with(|d| d.set(d.get() - 1));
        }
    }
}

thread_local! {
    /// How many compiled functions are running on this thread, one inside the
    /// other.
    static DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// The address the native stack of this thread must stay above: its end,
    /// plus room for what the deepest compiled function still calls.
    static STACK_FLOOR: usize = stack_end().map_or(0, |end| end + STACK_MARGIN);
}

const STACK_MARGIN: usize = 512 * 1024;

/// The lowest address of the calling thread's stack, where that is known.
#[cfg(target_os = "linux")]
fn stack_end() -> Option<usize> {
    extern "C" {
        fn pthread_self() -> usize;
        fn pthread_getattr_np(thread: usize, attr: *mut u64) -> i32;
        fn pthread_attr_getstack(attr: *const u64, addr: *mut *mut u8, size: *mut usize) -> i32;
        fn pthread_attr_destroy(attr: *mut u64) -> i32;
    }
    // Larger than pthread_attr_t on every Linux target.
    let mut attr = [0u64; 16];
    let (mut addr, mut size) = (null_mut(), 0usize);
    unsafe {
        if pthread_getattr_np(pthread_self(), attr.as_mut_ptr()) != 0 {
            return None;
        }
        let found = pthread_attr_getstack(attr.as_ptr(), &mut addr, &mut size) == 0;
        pthread_attr_destroy(attr.as_mut_ptr());
        found.then_some(addr as usize)
    }
}

#[cfg(not(target_os = "linux"))]
fn stack_end() -> Option<usize> {
    None
}

pub struct Compiled {
    pub callees: Arc<Callees>,
    global_cache: LoadCaches,
    module_attr_cache: LoadCaches,
    /// Per constant: usable as a cached module attribute name.
    plain_name: Vec<bool>,
    pub func: Function,
    pub analysis: Analysis,
    pub generic: Vec<Arc<BlockCode>>,
    settled: SettledBlocks,
    pub state: Shared<State>,
    pub options: Options,
    pub pipeline: Arc<Pipeline>,
    frames: Shared<Vec<Vec<Obj>>>,
    unbound: Unbound,
    /// Kernel results that were kept unboxed, and how many of those got a
    /// Python object after all.
    pub unboxed_results: AtomicU64,
    pub boxed_later: AtomicU64,
    /// Function name and file name of each origin, for traceback entries.
    origin_c: Vec<(CString, CString)>,
    /// Per handler block: the registers that survive entering it, which are
    /// what it reads and the frame slots.
    handler_keep: Vec<Option<analysis::BitSet>>,
    /// The values that are live at the instructions.
    live: Shared<HashMap<(usize, usize), Arc<analysis::BitSet>>>,
    /// The function without what its holds name, and its liveness: made
    /// when `read_at` is first asked.
    reads: std::sync::OnceLock<(Function, Analysis)>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Generic,
    Profile,
    Spec,
}

const MAX_CAPTURE_ATTEMPTS: u32 = 8;
/// How many profiling runs of an instruction that is no candidate for a call
/// site count its dispatcher calls.
const MAX_COUNTED_RUNS: u32 = 8;

type OpResult = Result<Obj, ()>;

unsafe fn set_error(exc: Obj, msg: String) {
    let c = CString::new(msg).unwrap_or_default();
    ffi::PyErr_SetString(exc, c.as_ptr());
}

/// What `len` gives for what a `__len__` returned, as the interpreter makes
/// it (`slot_sq_length`, then `len`): the index of it, which may not be
/// negative and has to fit, as an int. Takes the reference; null with the
/// error set where `len` raises.
unsafe fn counted(returned: Obj) -> Obj {
    let index = ffi::PyNumber_Index(returned);
    ffi::Py_DECREF(returned);
    if index.is_null() {
        return null_mut();
    }
    let zero = ffi::PyLong_FromLong(0);
    let negative = ffi::PyObject_RichCompareBool(index, zero, ffi::Py_LT);
    ffi::Py_DECREF(zero);
    if negative != 0 {
        if negative > 0 {
            set_error(ffi::PyExc_ValueError, "__len__() should return >= 0".to_string());
        }
        ffi::Py_DECREF(index);
        return null_mut();
    }
    let count = ffi::PyNumber_AsSsize_t(index, ffi::PyExc_OverflowError);
    ffi::Py_DECREF(index);
    if count == -1 && !ffi::PyErr_Occurred().is_null() {
        return null_mut();
    }
    ffi::PyLong_FromSsize_t(count)
}

/// What stands for an object whose `__iter__` returned `returned`, for a
/// builtin that goes over the object (`pyapi::Iterated`). Owned; null with
/// the error set.
unsafe fn iterated(returned: Obj) -> Obj {
    let py = pyo3::Python::assume_attached();
    let returned = pyo3::Bound::<pyo3::PyAny>::from_borrowed_ptr(py, returned).unbind();
    match pyo3::Py::new(py, crate::pyapi::Iterated { returned }) {
        Ok(stand) => stand.into_ptr(),
        Err(error) => {
            error.restore(py);
            null_mut()
        }
    }
}

/// A NameError carries the name it is about, as the interpreter's does (an
/// UnboundLocalError does not).
unsafe fn set_name_error(exc: Obj, msg: String, name: Obj) {
    set_error(exc, msg);
    let raised = ffi::PyErr_GetRaisedException();
    if !raised.is_null() {
        if !name.is_null() && ffi::PyObject_SetAttrString(raised, c"name".as_ptr(), name) < 0 {
            ffi::PyErr_Clear();
        }
        ffi::PyErr_SetRaisedException(raised);
    }
}

/// What the interpreter accepts after a `*`: something with `__iter__`, or a
/// sequence.
unsafe fn is_iterable(o: Obj) -> bool {
    (*ffi::Py_TYPE(o)).tp_iter.is_some() || ffi::PySequence_Check(o) != 0
}

#[inline(always)]
unsafe fn take(regs: &mut [Obj], v: V) -> Obj {
    std::mem::replace(regs.get_unchecked_mut(v as usize), null_mut())
}

/// The raw content of a register: a Python object, null, or an unboxed
/// tensor. Only for code that handles all three; see `Compiled::obj`.
#[inline(always)]
unsafe fn reg(regs: &[Obj], v: V) -> Obj {
    *regs.get_unchecked(v as usize)
}

#[inline(always)]
unsafe fn checked(obj: Obj) -> OpResult {
    if obj.is_null() {
        Err(())
    } else {
        Ok(obj)
    }
}

#[inline(always)]
unsafe fn status(rc: i32) -> OpResult {
    if rc < 0 {
        Err(())
    } else {
        Ok(null_mut())
    }
}

unsafe fn binary(kind: u8, a: Obj, b: Obj) -> Obj {
    match kind {
        0 => ffi::PyNumber_Add(a, b),
        1 => ffi::PyNumber_And(a, b),
        2 => ffi::PyNumber_FloorDivide(a, b),
        3 => ffi::PyNumber_Lshift(a, b),
        4 => ffi::PyNumber_MatrixMultiply(a, b),
        5 => ffi::PyNumber_Multiply(a, b),
        6 => ffi::PyNumber_Remainder(a, b),
        7 => ffi::PyNumber_Or(a, b),
        8 => ffi::PyNumber_Power(a, b, ffi::Py_None()),
        9 => ffi::PyNumber_Rshift(a, b),
        10 => ffi::PyNumber_Subtract(a, b),
        11 => ffi::PyNumber_TrueDivide(a, b),
        12 => ffi::PyNumber_Xor(a, b),
        13 => ffi::PyNumber_InPlaceAdd(a, b),
        14 => ffi::PyNumber_InPlaceAnd(a, b),
        15 => ffi::PyNumber_InPlaceFloorDivide(a, b),
        16 => ffi::PyNumber_InPlaceLshift(a, b),
        17 => ffi::PyNumber_InPlaceMatrixMultiply(a, b),
        18 => ffi::PyNumber_InPlaceMultiply(a, b),
        19 => ffi::PyNumber_InPlaceRemainder(a, b),
        20 => ffi::PyNumber_InPlaceOr(a, b),
        21 => ffi::PyNumber_InPlacePower(a, b, ffi::Py_None()),
        22 => ffi::PyNumber_InPlaceRshift(a, b),
        23 => ffi::PyNumber_InPlaceSubtract(a, b),
        24 => ffi::PyNumber_InPlaceTrueDivide(a, b),
        _ => ffi::PyNumber_InPlaceXor(a, b),
    }
}

/// `raise exc from cause`, as the interpreter does it.
unsafe fn do_raise(exc: Obj, cause: Obj) {
    let (ty, value) = if ffi::PyExceptionClass_Check(exc) != 0 {
        let value = ffi::PyObject_CallNoArgs(exc);
        if value.is_null() {
            return;
        }
        if ffi::PyExceptionInstance_Check(value) == 0 {
            set_error(
                ffi::PyExc_TypeError,
                "calling an exception class did not return an exception".to_string(),
            );
            ffi::Py_DECREF(value);
            return;
        }
        (exc, value)
    } else if ffi::PyExceptionInstance_Check(exc) != 0 {
        (ffi::Py_TYPE(exc) as Obj, new_ref(exc))
    } else {
        set_error(ffi::PyExc_TypeError, "exceptions must derive from BaseException".to_string());
        return;
    };
    if !cause.is_null() {
        let fixed = if ffi::PyExceptionClass_Check(cause) != 0 {
            let c = ffi::PyObject_CallNoArgs(cause);
            if c.is_null() {
                ffi::Py_DECREF(value);
                return;
            }
            c
        } else if ffi::PyExceptionInstance_Check(cause) != 0 {
            new_ref(cause)
        } else if cause == ffi::Py_None() {
            null_mut()
        } else {
            set_error(
                ffi::PyExc_TypeError,
                "exception causes must derive from BaseException".to_string(),
            );
            ffi::Py_DECREF(value);
            return;
        };
        ffi::PyException_SetCause(value, fixed);
    }
    ffi::PyErr_SetObject(ty, value);
    ffi::Py_DECREF(value);
}

impl Compiled {
    /// The value of a register as a Python object. A kernel result is kept
    /// unboxed until this is called on it, that is until generic code looks
    /// at it; then it gets its Python object, in place.
    #[inline(always)]
    unsafe fn obj(&self, regs: &mut [Obj], v: V) -> Obj {
        let o = reg(regs, v);
        if capi::is_unboxed(o) {
            self.box_register(regs, v)
        } else {
            o
        }
    }

    #[cold]
    unsafe fn box_register(&self, regs: &mut [Obj], v: V) -> Obj {
        let boxed = (capi::api().unwrap_unchecked().tensor_box)(reg(regs, v));
        if boxed.is_null() {
            // The reference was consumed either way.
            *regs.get_unchecked_mut(v as usize) = null_mut();
            panic!("torch.vm: could not create the Python object of a tensor");
        }
        *regs.get_unchecked_mut(v as usize) = boxed;
        self.boxed_later.fetch_add(1, Ordering::Relaxed);
        boxed
    }

    /// Binds a variable of `frame` to the content of a register, or unbinds
    /// it for an empty one (see `Op::StoreLocal`). A tensor without a Python
    /// object does not get one for this: the variable stays unbound until
    /// `bind` runs.
    #[inline(always)]
    unsafe fn store_local(&self, frame: Frame, slot: u32, o: Obj) {
        if frame.locals.is_null() {
            return;
        }
        let place = frame.locals.add(slot as usize);
        if self.unbound.any.get() {
            self.forget(|entry| entry.1 == place);
        }
        let bound = if capi::is_unboxed(o) {
            self.wait(frame, place, o);
            null_mut()
        } else {
            ffi::Py_XINCREF(o);
            o
        };
        // The slot first: letting go of what it held can run a finalizer.
        xdecref(frame_owned(std::mem::replace(&mut *place, frame_ref(bound))));
    }

    #[inline(never)]
    unsafe fn wait(&self, frame: Frame, place: *mut Obj, tensor: Obj) {
        capi::retain(tensor);
        self.unbound.list.borrow_mut().push((frame.ptr, place, tensor));
        self.unbound.any.set(true);
    }

    /// Gives the tensors that variables are bound to their Python objects,
    /// and the frames the objects. Leaves the error indicator as it is.
    #[cold]
    unsafe fn bind(&self) {
        let list = std::mem::take(&mut *self.unbound.list.borrow_mut());
        self.unbound.any.set(false);
        let raised = ffi::PyErr_GetRaisedException();
        let api = capi::api().unwrap_unchecked();
        for (_, place, tensor) in list {
            // Consumes the reference.
            let boxed = (api.tensor_box)(tensor);
            if boxed.is_null() {
                ffi::PyErr_Clear();
            } else {
                xdecref(frame_owned(std::mem::replace(&mut *place, frame_ref(boxed))));
            }
        }
        if !raised.is_null() {
            ffi::PyErr_SetRaisedException(raised);
        }
    }

    /// Lets go of the tensors of the variables `which` picks, which stay
    /// unbound.
    #[cold]
    unsafe fn forget(&self, which: impl Fn(&(*mut std::ffi::c_void, *mut Obj, Obj)) -> bool) {
        let mut list = self.unbound.list.borrow_mut();
        let mut i = 0;
        while i < list.len() {
            if which(&list[i]) {
                capi::release(list.swap_remove(i).2);
            } else {
                i += 1;
            }
        }
        self.unbound.any.set(!list.is_empty());
    }

    /// For a frame that is about to be popped. What holds on to the frame
    /// (the traceback of an exception, as a rule) finds its variables bound.
    #[inline(always)]
    unsafe fn settle(&self, frame: Frame) {
        if self.unbound.any.get() {
            if frame.held() {
                self.bind();
            } else {
                self.forget(|entry| entry.0 == frame.ptr);
            }
        }
    }

    /// Pops the frame of the inlined function that is running.
    unsafe fn leave(&self, frames: &mut Frames) {
        if frames.depth > 0 {
            self.settle(frames.top());
        }
        frames.leave();
    }

    pub fn new(
        mut func: Function,
        options: Options,
        pipeline: Arc<Pipeline>,
        callees: Arc<Callees>,
    ) -> Compiled {
        // One value no instruction of the function defines: where a pass can
        // put a result it takes apart right away (see `Region.fuse`).
        func.value_names.push(None);
        func.nvalues += 1;
        let analysis = analysis::analyze(&func);
        let mut generic = Vec::with_capacity(func.blocks.len());
        for (b, block) in func.blocks.iter().enumerate() {
            let mut insts = block.insts.clone();
            let term_kills =
                analysis::compute_kills(&mut insts, &block.term, &analysis.live_out[b]);
            generic.push(Arc::new(BlockCode {
                insts,
                term: block.term.clone(),
                term_kills,
                entry_guards: Vec::new(),
                id: 0,
                origins: (0..block.insts.len() as u32).collect(),
                graphs: Vec::new(),
                left: Vec::new(),
                entering: None,
            }));
        }
        let blocks = func
            .blocks
            .iter()
            .map(|block| BlockState {
                versions: Vec::new(),
                retired: Vec::new(),
                slots: vec![None; block.insts.len()],
                reasons: vec![None; block.insts.len()],
                capture_attempts: vec![0; block.insts.len()],
                facts: vec![InstFacts::default(); block.insts.len()],
                respec_left: options.max_respecializations,
                profile_runs: 0,
                raised: 0,
                next_version_id: 1,
                execs: 0,
                spec_execs: 0,
                entry_misses: 0,
                deopts: 0,
                rejected: None,
                note: None,
                graphs: Vec::new(),
            })
            .collect();
        let state = State {
            blocks,
            obs: vec![ValueObs::Unseen; func.nvalues as usize],
            calls: 0,
            log: Vec::new(),
        };
        let empty = || LoadCaches(vec![std::cell::Cell::new(LoadCache::EMPTY); func.consts.len()]);
        let plain_name = func
            .consts
            .iter()
            .map(|c| unsafe {
                ffi::PyUnicode_CheckExact(c.as_ptr()) != 0
                    && !pyffi::str_to_string(c.as_ptr()).starts_with("__")
            })
            .collect();
        let mut handler_keep: Vec<Option<analysis::BitSet>> = vec![None; func.blocks.len()];
        for block in &func.blocks {
            if block.handler != NONE && handler_keep[block.handler as usize].is_none() {
                let mut keep = analysis.live_in[block.handler as usize].clone();
                for slot in &func.slots {
                    keep.insert(*slot);
                }
                handler_keep[block.handler as usize] = Some(keep);
            }
        }
        Compiled {
            callees,
            handler_keep,
            live: Shared::new(HashMap::new()),
            reads: std::sync::OnceLock::new(),
            global_cache: empty(),
            module_attr_cache: empty(),
            plain_name,
            origin_c: func
                .origins
                .iter()
                .map(|o| {
                    (
                        CString::new(o.name.clone()).unwrap_or_default(),
                        CString::new(o.filename.clone()).unwrap_or_default(),
                    )
                })
                .collect(),
            func,
            analysis,
            settled: SettledBlocks {
                blocks: (0..generic.len()).map(|_| std::cell::Cell::new(Settled { code: std::ptr::null(), specialized: false, entries: 0 })).collect(),
                warming: std::cell::Cell::new(0),
                calls: std::cell::Cell::new(0),
                trial: std::cell::Cell::new(Trial::start(PLAN_EPOCH.load(Ordering::Relaxed))),
                watched: std::cell::Cell::new(false),
                below: std::cell::Cell::new(false),
            },
            generic,
            state: Shared::new(state),
            options,
            pipeline,
            frames: Shared::new(Vec::new()),
            unbound: Unbound { list: std::cell::RefCell::new(Vec::new()), any: std::cell::Cell::new(false) },
            unboxed_results: AtomicU64::new(0),
            boxed_later: AtomicU64::new(0),
        }
    }

    // ------------------------------------------------------------------
    // Calling convention
    // ------------------------------------------------------------------

    unsafe fn bind_args(&self, regs: &mut [Obj], function: Obj, args: &Args, kwargs: Obj) -> Result<(), ()> {
        let f = &self.func;
        let nargs = args.len();
        let argcount = f.argcount;
        let nkw = if kwargs.is_null() { 0 } else { ffi::PyDict_Size(kwargs) as usize };
        for i in 0..nargs.min(argcount) {
            regs[f.params[i] as usize] = new_ref(args.get(i));
        }
        let mut next = argcount + f.kwonlyargcount;
        if f.has_varargs {
            let rest = args.slice(argcount.min(nargs));
            if rest.is_null() {
                return Err(());
            }
            regs[f.params[next] as usize] = rest;
            next += 1;
        } else if nargs > argcount {
            set_error(
                ffi::PyExc_TypeError,
                format!(
                    "{}() takes {} positional argument{} but {} were given",
                    f.name,
                    argcount,
                    if argcount == 1 { "" } else { "s" },
                    nargs
                ),
            );
            return Err(());
        }
        let mut varkw: Obj = null_mut();
        if f.has_varkw {
            varkw = ffi::PyDict_New();
            if varkw.is_null() {
                return Err(());
            }
            regs[f.params[next] as usize] = varkw;
        }
        if nkw > 0 {
            let mut pos: ffi::Py_ssize_t = 0;
            let mut key: Obj = null_mut();
            let mut value: Obj = null_mut();
            while ffi::PyDict_Next(kwargs, &mut pos, &mut key, &mut value) != 0 {
                let mut found = None;
                for j in f.posonlyargcount..argcount + f.kwonlyargcount {
                    if f.param_names[j].as_ptr() == key {
                        found = Some(j);
                        break;
                    }
                }
                if found.is_none() {
                    for j in f.posonlyargcount..argcount + f.kwonlyargcount {
                        let eq = ffi::PyObject_RichCompareBool(
                            f.param_names[j].as_ptr(),
                            key,
                            ffi::Py_EQ,
                        );
                        if eq < 0 {
                            return Err(());
                        }
                        if eq > 0 {
                            found = Some(j);
                            break;
                        }
                    }
                }
                match found {
                    Some(j) => {
                        let slot = &mut regs[f.params[j] as usize];
                        if !slot.is_null() {
                            set_error(
                                ffi::PyExc_TypeError,
                                format!(
                                    "{}() got multiple values for argument '{}'",
                                    f.name,
                                    pyffi::str_to_string(key)
                                ),
                            );
                            return Err(());
                        }
                        *slot = new_ref(value);
                    }
                    None if !varkw.is_null() => {
                        if ffi::PyDict_SetItem(varkw, key, value) < 0 {
                            return Err(());
                        }
                    }
                    None => {
                        set_error(
                            ffi::PyExc_TypeError,
                            format!(
                                "{}() got an unexpected keyword argument '{}'",
                                f.name,
                                pyffi::str_to_string(key)
                            ),
                        );
                        return Err(());
                    }
                }
            }
        }
        if nargs < argcount {
            let defaults = ffi::PyFunction_GetDefaults(function);
            let ndefaults = if defaults.is_null() { 0 } else { ffi::PyTuple_GET_SIZE(defaults) as usize };
            for i in nargs..argcount {
                let slot = &mut regs[f.params[i] as usize];
                if !slot.is_null() {
                    continue;
                }
                if i + ndefaults >= argcount {
                    *slot = new_ref(ffi::PyTuple_GET_ITEM(defaults, (i + ndefaults - argcount) as isize));
                } else {
                    set_error(
                        ffi::PyExc_TypeError,
                        format!(
                            "{}() missing 1 required positional argument: '{}'",
                            f.name,
                            pyffi::str_to_string(f.param_names[i].as_ptr())
                        ),
                    );
                    return Err(());
                }
            }
        }
        for j in argcount..argcount + f.kwonlyargcount {
            let slot = &mut regs[f.params[j] as usize];
            if !slot.is_null() {
                continue;
            }
            let mut default: Obj = null_mut();
            let kwdefaults = ffi::PyFunction_GetKwDefaults(function);
            if !kwdefaults.is_null() {
                default = ffi::PyDict_GetItemWithError(kwdefaults, f.param_names[j].as_ptr());
                if default.is_null() && !ffi::PyErr_Occurred().is_null() {
                    return Err(());
                }
            }
            if default.is_null() {
                set_error(
                    ffi::PyExc_TypeError,
                    format!(
                        "{}() missing 1 required keyword-only argument: '{}'",
                        f.name,
                        pyffi::str_to_string(f.param_names[j].as_ptr())
                    ),
                );
                return Err(());
            }
            *slot = new_ref(default);
        }
        Ok(())
    }

    /// Calls `function`, a Python function with the code and globals this
    /// was compiled from: its defaults and its closure are read from it.
    /// `args` is a tuple, `kwargs` a dict or null. Returns a new reference,
    /// or null with a Python error set.
    pub unsafe fn call(&self, function: Obj, args: Obj, kwargs: Obj) -> Obj {
        self.enter(function, Args::tuple(null_mut(), args), kwargs)
    }

    /// `call` with the positional arguments wherever the caller has them.
    unsafe fn enter(&self, function: Obj, args: Args, kwargs: Obj) -> Obj {
        // A trace function (a debugger, a coverage tool) is told about the
        // frames that are entered from now on and the lines that run in
        // them: the interpreter runs those. A profile function is not
        // looked at: what it would time is the interpreter then.
        if capi::traced() {
            let all = args.slice(0);
            if all.is_null() {
                return null_mut();
            }
            let result = ffi::PyObject_Call(function, all, kwargs);
            ffi::Py_DECREF(all);
            return result;
        }
        // Unlike a call the interpreter makes, this one uses native stack,
        // about 2 kB a level, so the stack itself is checked. Where compiled
        // functions have frames, pushing one counts the call against the
        // recursion limit in the interpreter's own counter; elsewhere the VM
        // counts, and Py_EnterRecursiveCall counts the C call.
        let framed = capi::has_frames();
        let here = std::ptr::addr_of!(framed) as usize;
        let mut too_deep = here < STACK_FLOOR.with(|floor| *floor);
        if !framed && !too_deep {
            let depth = DEPTH.with(|d| {
                d.set(d.get() + 1);
                d.get()
            });
            too_deep = depth > ffi::Py_GetRecursionLimit() as usize;
            if !too_deep
                && ffi::Py_EnterRecursiveCall(c" while calling a cfg-compiled function".as_ptr()) != 0
            {
                DEPTH.with(|d| d.set(d.get() - 1));
                return null_mut();
            }
            if too_deep {
                DEPTH.with(|d| d.set(d.get() - 1));
            }
        }
        if too_deep {
            set_error(ffi::PyExc_RecursionError, "maximum recursion depth exceeded".to_string());
            return null_mut();
        }
        // One register past the values holds the function for the length of
        // the call, as a frame of the interpreter does.
        let nregs = self.func.nvalues as usize + 1;
        let mut regs = self.frames.borrow_mut().pop().unwrap_or_else(|| vec![null_mut(); nregs]);
        regs[nregs - 1] = new_ref(function);
        let result = match self.bind_args(&mut regs, function, &args, kwargs) {
            Ok(()) => match Frame::push(function) {
                Ok(frame) => {
                    // The frame starts out with the arguments, as the
                    // interpreter's does; the function binds the rest.
                    if !frame.locals.is_null() {
                        for (slot, v) in self.func.params.iter().enumerate() {
                            let o = reg(&regs, *v);
                            if !o.is_null() {
                                *frame.locals.add(slot) = frame_ref(new_ref(o));
                            }
                        }
                    }
                    self.run(&mut regs, frame)
                }
                Err(()) => null_mut(),
            },
            // Arguments that do not fit the signature: the interpreter turns
            // the call down before it runs anything, in its own words.
            Err(()) => {
                ffi::PyErr_Clear();
                let all = args.slice(0);
                if all.is_null() {
                    null_mut()
                } else {
                    let result = ffi::PyObject_Call(function, all, kwargs);
                    ffi::Py_DECREF(all);
                    result
                }
            }
        };
        for r in regs.iter_mut() {
            capi::release(std::mem::replace(r, null_mut()));
        }
        self.frames.borrow_mut().push(regs);
        if !framed {
            ffi::Py_LeaveRecursiveCall();
            DEPTH.with(|d| d.set(d.get() - 1));
        }
        result
    }

    /// Runs the function in a frame that has its arguments, and pops the
    /// frame.
    #[inline(always)]
    unsafe fn run(&self, regs: &mut [Obj], frame: Frame) -> Obj {
        let result = self.exec(regs, frame);
        self.returned(frame, result)
    }

    /// What is done when the function has run in a frame: pops the frame.
    #[inline(always)]
    unsafe fn returned(&self, frame: Frame, result: Obj) -> Obj {
        let mut result = result;
        // A debugger that was entered below this call has its trace function
        // on the frame, and is told that the function returns.
        if !frame.object.is_null() && !(*frame.object).is_null() && !result.is_null() {
            if let Some(returns) = capi::api().and_then(|api| api.frame_returns) {
                if returns(frame.ptr, result) != 0 {
                    ffi::Py_DECREF(result);
                    result = null_mut();
                }
            }
        }
        // The traceback of an exception that leaves the function refers to
        // its frame, which takes its variables along.
        self.settle(frame);
        if self.unbound.any.get() {
            self.forget(|_| true);
        }
        // Before the registers are released, as the interpreter unlinks a
        // frame before it clears its locals.
        frame.pop();
        result
    }

    /// `enter` for a call the interpreter has made a frame for (see
    /// `take_frame`): the arguments are bound in it, as the variables they
    /// are, and the function runs in it. The frame is gone on return.
    unsafe fn enter_frame(&self, function: Obj, made: *mut std::ffi::c_void) -> Obj {
        let api = capi::api().unwrap_unchecked();
        let mut frame = Frame { ptr: made, ..Frame::NONE };
        let adopt = api.frame_adopt.unwrap_unchecked();
        if adopt(made, &mut frame.instr, &mut frame.code, &mut frame.locals, &mut frame.object) == 0 {
            return null_mut();
        }
        let here = std::ptr::addr_of!(frame) as usize;
        if here < STACK_FLOOR.with(|floor| *floor) {
            set_error(ffi::PyExc_RecursionError, "maximum recursion depth exceeded".to_string());
            frame.pop();
            return null_mut();
        }
        // The call may come from any instruction of this same function (an
        // attribute that is a property, an operator), not only from a call,
        // before which the variables of its frames have their objects.
        if self.unbound.any.get() {
            self.bind();
        }
        let nregs = self.func.nvalues as usize + 1;
        let mut regs = self.frames.borrow_mut().pop().unwrap_or_else(|| vec![null_mut(); nregs]);
        regs[nregs - 1] = new_ref(function);
        for (slot, v) in self.func.params.iter().enumerate() {
            let o = frame_value(*frame.locals.add(slot));
            if !o.is_null() {
                regs[*v as usize] = new_ref(o);
            }
        }
        let result = self.run(&mut regs, frame);
        for r in regs.iter_mut() {
            capi::release(std::mem::replace(r, null_mut()));
        }
        self.frames.borrow_mut().push(regs);
        result
    }

    /// Hands the exception that was raised in block `cur` to the block's
    /// handler and returns that block; None if the block has no handler and
    /// the exception leaves the function.
    unsafe fn catch(&self, regs: &mut [Obj], cur: usize) -> Option<usize> {
        let handler = self.func.blocks.get_unchecked(cur).handler;
        if handler == NONE {
            return None;
        }
        let exc = ffi::PyErr_GetRaisedException();
        if exc.is_null() {
            return None;
        }
        // The handler's phis, for the edge from this block.
        let block = self.func.blocks.get_unchecked(handler as usize);
        if !block.phis.is_empty() {
            let pi = block.preds.iter().position(|p| *p as usize == cur).unwrap_unchecked();
            let moves = self.analysis.edge_moves.get_unchecked(handler as usize).get_unchecked(pi);
            let vals: Vec<Obj> = moves
                .iter()
                .map(|(_, src)| {
                    let o = reg(regs, *src);
                    capi::retain(o);
                    o
                })
                .collect();
            for ((dst, _), val) in moves.iter().zip(vals) {
                capi::release(std::mem::replace(regs.get_unchecked_mut(*dst as usize), val));
            }
        }
        // Everything the handler does not read is dead from here on.
        let keep = self.handler_keep.get_unchecked(handler as usize).as_ref().unwrap_unchecked();
        let nvalues = self.func.nvalues as usize;
        for (v, r) in regs[..nvalues].iter_mut().enumerate() {
            if !r.is_null() && !keep.contains(v as V) {
                capi::release(std::mem::replace(r, null_mut()));
            }
        }
        *regs.get_unchecked_mut(self.func.exc_reg as usize) = exc;
        Some(handler as usize)
    }

    /// Records the error that the instruction at `bc` raised in the
    /// traceback: one entry per frame the interpreter would have had, the
    /// inlined function the instruction came from and each call it sits
    /// under, and unwinds the frames of the inlined functions. Where frames
    /// are supported the entries come from them, which gives them the
    /// position within the line.
    unsafe fn fail(&self, frames: &mut Frames, bc: u32) -> Obj {
        let mut bc = bc;
        // The traceback takes the frames along, and they their variables.
        if self.unbound.any.get() {
            self.bind();
        }
        loop {
            let origin = (bc >> ORIGIN_SHIFT) as usize;
            let frame = frames.top();
            if frame.ptr.is_null() {
                let (name, filename) = &self.origin_c[origin];
                pyffi::_PyTraceback_Add(name.as_ptr(), filename.as_ptr(), self.func.line_of(bc));
            } else {
                frame.at(bc & OFFSET_MASK);
                let current = ffi::PyEval_GetFrame();
                if !current.is_null() {
                    ffi::PyTraceBack_Here(current);
                }
            }
            if origin == 0 {
                break;
            }
            self.leave(frames);
            bc = self.func.origins[origin].call_bc;
        }
        null_mut()
    }

    // ------------------------------------------------------------------
    // Generic instructions
    // ------------------------------------------------------------------

    unsafe fn vectorcall(&self, regs: &mut [Obj], callee: Obj, args: &[V], kwnames: u32) -> Obj {
        // One spare slot in front lets bound methods prepend self in place.
        let mut small: [Obj; 9] = [null_mut(); 9];
        let mut big: Vec<Obj>;
        let buf: &mut [Obj] = if args.len() < small.len() {
            &mut small[..]
        } else {
            big = vec![null_mut(); args.len() + 1];
            &mut big[..]
        };
        for (i, v) in args.iter().enumerate() {
            buf[i + 1] = self.obj(regs, *v);
        }
        let (kw, nkw) = if kwnames == NONE {
            (null_mut(), 0)
        } else {
            let kw = self.func.consts[kwnames as usize].as_ptr();
            (kw, ffi::PyTuple_GET_SIZE(kw) as usize)
        };
        let npos = args.len() - nkw;
        let interpreted = || {
            ffi::PyObject_Vectorcall(
                callee,
                buf.as_ptr().add(1),
                npos | ffi::PY_VECTORCALL_ARGUMENTS_OFFSET,
                kw,
            )
        };
        match self.compiled_callee(callee) {
            Some((compiled, func, receiver)) => self.tiered(
                compiled,
                func,
                || self.call_compiled(&*compiled, func, receiver, &buf[1..args.len() + 1], kw),
                interpreted,
            ),
            None => interpreted(),
        }
    }

    /// The compiled form of what calling `callee` runs, if it is a Python
    /// function the frontend supports, with that function and the object it
    /// is bound to (null for a plain function): a function, a bound method,
    /// or an instance of a class whose `__call__` is a Python function.
    unsafe fn compiled_callee(&self, callee: Obj) -> Option<(*const Compiled, Obj, Obj)> {
        // The call the profiling tier is looking at is this one, whatever it
        // comes down to.
        let profiled = self.callees.callee_asked.get();
        if profiled {
            self.callees.callee_asked.set(false);
        }
        if !self.options.callees || REPLAY_DEPTH.with(|d| d.get()) > 0 {
            return None;
        }
        let (func, receiver, kind) = self.target(callee)?;
        if profiled {
            self.note_callee(func, kind);
        }
        self.compiled_function(func).map(|compiled| (compiled, func, receiver))
    }

    /// What calling `callee` comes down to, if that is a Python function:
    /// the function, the object it is bound to (null for a plain function),
    /// and which of the three kinds of callee it is.
    #[inline(always)]
    unsafe fn target(&self, callee: Obj) -> Option<(Obj, Obj, CalleeKind)> {
        let (func, receiver, kind) = if ffi::PyFunction_Check(callee) != 0 {
            (callee, null_mut(), CalleeKind::Function)
        } else if ffi::Py_TYPE(callee) == std::ptr::addr_of_mut!(pyffi::PyMethod_Type) {
            (pyffi::PyMethod_Function(callee), pyffi::PyMethod_Self(callee), CalleeKind::Method)
        } else {
            let ty = ffi::Py_TYPE(callee);
            if ffi::PyType_HasFeature(ty, ffi::Py_TPFLAGS_HEAPTYPE) == 0 || ffi::PyType_Check(callee) != 0 {
                return None;
            }
            let call = pyffi::_PyType_Lookup(ty, self.callees.call_name.as_ptr());
            // What the call comes down to, if a shortcut says so: for a
            // module without hooks, its forward.
            let target = if self.options.shortcuts && !call.is_null() {
                crate::shortcuts::call_target(callee, ty, call)
            } else {
                null_mut()
            };
            (if target.is_null() { call } else { target }, callee, CalleeKind::Instance)
        };
        if func.is_null() || ffi::PyFunction_Check(func) == 0 {
            return None;
        }
        Some((func, receiver, kind))
    }

    #[cold]
    unsafe fn note_callee(&self, func: Obj, kind: CalleeKind) {
        let code = PyObj::from_borrowed(ffi::PyFunction_GetCode(func));
        *self.callees.callee_seen.borrow_mut() = Some((code, kind));
    }

    #[inline(always)]
    unsafe fn compiled_function(&self, func: Obj) -> Option<*const Compiled> {
        let form = self.function_form(func);
        if form == INTERPRETED || form == DEMOTED {
            None
        } else {
            Some(form)
        }
    }

    /// What a Python function runs as: its compiled form, or one of
    /// INTERPRETED and DEMOTED.
    #[inline(always)]
    unsafe fn function_form(&self, func: Obj) -> *const Compiled {
        let callees = &self.callees;
        let code = ffi::PyFunction_GetCode(func);
        let globals = ffi::PyFunction_GetGlobals(func);
        let slot = callees.recent.get_unchecked((code as usize >> 4) & (RECENT_CALLEES - 1));
        let (recent_code, recent_globals, recent) = slot.get();
        if recent_code == code as usize && recent_globals == globals as usize {
            return recent;
        }
        let form = self.function_form_slow(func, code, globals);
        slot.set((code as usize, globals as usize, form));
        form
    }

    unsafe fn function_form_slow(&self, func: Obj, code: Obj, globals: Obj) -> *const Compiled {
        let callees = &self.callees;
        let key = (code as usize, globals as usize);
        if let Some(c) = callees.entries.borrow().get(&key) {
            let compiled = Arc::as_ptr(&c.compiled);
            return if (*compiled).settled.trial.get().phase == TRIAL_DEMOTED { DEMOTED } else { compiled };
        }
        if callees.left.borrow().contains_key(&(code as usize)) {
            return DEMOTED;
        }
        if callees.interpreted.borrow().contains_key(&(code as usize))
            || callees.entries.borrow().len() >= MAX_CALLEES
        {
            return INTERPRETED;
        }
        if is_left(code) {
            callees.left.borrow_mut().insert(code as usize, PyObj::from_borrowed(code));
            return DEMOTED;
        }
        let py = pyo3::Python::assume_attached();
        let bound = pyo3::Bound::from_borrowed_ptr(py, func);
        let _nobody = Hooked::nobody();
        match lower(py, &callees.table, &bound, &self.options) {
            Ok(function) => {
                let compiled = Arc::new(Compiled::new(
                    function,
                    self.options.clone(),
                    self.pipeline.clone(),
                    callees.clone(),
                ));
                let pointer = Arc::as_ptr(&compiled);
                callees.entries.borrow_mut().insert(
                    key,
                    Callee {
                        code: PyObj::from_borrowed(code),
                        _globals: PyObj::from_borrowed(globals),
                        compiled,
                    },
                );
                pointer
            }
            // Unsupported by the frontend: the interpreter runs it.
            Err(_) => {
                callees.interpreted.borrow_mut().insert(code as usize, PyObj::from_borrowed(code));
                INTERPRETED
            }
        }
    }

    /// The value that no instruction of the function defines.
    pub fn scratch(&self) -> V {
        self.func.nvalues - 1
    }

    /// The values that are needed from instruction `index` of a block on.
    pub fn live_at(&self, block: usize, index: usize) -> Arc<analysis::BitSet> {
        if let Some(live) = self.live.borrow().get(&(block, index)) {
            return live.clone();
        }
        let live = Arc::new(self.analysis.live_before(&self.func, block, index));
        self.live.borrow_mut().insert((block, index), live.clone());
        live
    }

    /// The values that an instruction from `index` of a block on reads: the
    /// ones that are needed there less those that are only kept alive (see
    /// `Op::Hold`), which matters to something that would have to make them.
    pub fn read_at(&self, block: usize, index: usize) -> analysis::BitSet {
        let (func, analysis) = self.without_holds();
        analysis.live_before(func, block, index)
    }

    /// The values that the blocks after a block read.
    pub fn read_out(&self, block: usize) -> analysis::BitSet {
        self.without_holds().1.live_out[block].clone()
    }

    fn without_holds(&self) -> &(Function, Analysis) {
        self.reads.get_or_init(|| {
            let mut func = self.func.clone();
            for found in func.blocks.iter_mut() {
                for inst in found.insts.iter_mut() {
                    if let Op::Hold(values) = &mut inst.op {
                        values.clear();
                    }
                }
            }
            let analysis = analysis::analyze(&func);
            (func, analysis)
        })
    }

    /// The values a range of the generic instructions of a block takes from
    /// before it, and the ones it defines that are needed after it.
    pub fn range_io(&self, block: usize, lo: usize, hi: usize) -> (Vec<V>, Vec<V>) {
        let insts = &self.func.blocks[block].insts;
        let mut defined: Vec<V> = Vec::new();
        let mut inputs: Vec<V> = Vec::new();
        for inst in &insts[lo..hi] {
            // A hold names values that may not exist yet (a local before its
            // first assignment), so it does not make a value an input; nor
            // does binding a variable, which is done after the range.
            if !matches!(inst.op, Op::Hold(_) | Op::StoreLocal(..)) {
                inst.op.for_each_use(|v| {
                    if v != UNDEF && v != NONE && !defined.contains(&v) && !inputs.contains(&v) {
                        inputs.push(v);
                    }
                });
            }
            if inst.dst != NONE {
                defined.push(inst.dst);
            }
        }
        let mut needed: Vec<V> = Vec::new();
        let mut need = |v: V| {
            if defined.contains(&v) && !needed.contains(&v) {
                needed.push(v);
            }
        };
        for inst in &insts[hi..] {
            inst.op.for_each_use(&mut need);
        }
        self.func.blocks[block].term.for_each_use(&mut need);
        for v in self.analysis.live_out[block].iter() {
            need(v);
        }
        // In the order they are defined.
        let outputs = defined.into_iter().filter(|v| needed.contains(v)).collect();
        (inputs, outputs)
    }

    /// Runs the generic instructions lo..hi of a block on the values in
    /// `env`, a dict from value numbers to objects, which afterwards holds
    /// the values that are still needed. With `traced`, what the
    /// instructions call is left to the interpreter. On an error returns the
    /// index of the instruction that raised it, with the error set.
    pub unsafe fn replay(&self, block: usize, lo: usize, hi: usize, env: Obj, traced: bool) -> Result<(), usize> {
        let insts = &self.generic[block].insts;
        let nvalues = self.func.nvalues as usize;
        let mut regs: Vec<Obj> = vec![null_mut(); nvalues + 1];
        let mut pos: ffi::Py_ssize_t = 0;
        let (mut key, mut value): (Obj, Obj) = (null_mut(), null_mut());
        while ffi::PyDict_Next(env, &mut pos, &mut key, &mut value) != 0 {
            let v = ffi::PyLong_AsSsize_t(key);
            if v >= 0 && (v as usize) < nvalues {
                regs[v as usize] = new_ref(value);
            }
        }
        ffi::PyErr_Clear();
        if traced {
            REPLAY_DEPTH.with(|d| d.set(d.get() + 1));
        }
        // The instructions run as they would in the function: called by its
        // frame, not by the frames of whatever asked for the replay, so that
        // what looks at its caller (`globals`, `sys._getframe`, a warning)
        // finds the function, at the line of the instruction.
        let frame = REGION_FRAME.with(|f| f.get());
        let swap = if frame.ptr.is_null() { None } else { capi::api().and_then(|api| api.frame_swap) };
        let outer = swap.map(|swap| swap(frame.ptr));
        let origin = insts.get(lo).map_or(0, |inst| inst.bc >> ORIGIN_SHIFT);
        let mut failed = None;
        for (i, inst) in insts.iter().enumerate().take(hi).skip(lo) {
            if !inst.op.is_replayable() {
                set_error(ffi::PyExc_RuntimeError, "the instruction cannot be replayed".to_string());
                failed = Some(i);
                break;
            }
            if outer.is_some() && inst.bc >> ORIGIN_SHIFT == origin {
                frame.at(inst.bc & OFFSET_MASK);
                // On the values of the call itself the variables are bound
                // as the function would bind them; a trace has stand-ins.
                if let (Op::StoreLocal(slot, v), false) = (&inst.op, traced) {
                    self.store_local(frame, *slot, reg(&regs, *v));
                }
            }
            match self.exec_op(&mut regs, &inst.op) {
                Ok(obj) => {
                    if inst.dst != NONE {
                        capi::release(std::mem::replace(&mut regs[inst.dst as usize], obj));
                    } else {
                        capi::release(obj);
                    }
                    for v in &inst.kills {
                        capi::release(take(&mut regs, *v));
                    }
                }
                Err(()) => {
                    failed = Some(i);
                    break;
                }
            }
        }
        if let (Some(swap), Some(outer)) = (swap, outer) {
            swap(outer);
        }
        if traced {
            REPLAY_DEPTH.with(|d| d.set(d.get() - 1));
        }
        let raised = if failed.is_some() { ffi::PyErr_GetRaisedException() } else { null_mut() };
        ffi::PyDict_Clear(env);
        for v in 0..nvalues {
            if regs[v].is_null() {
                continue;
            }
            if failed.is_none() {
                let object = self.obj(&mut regs, v as V);
                let key = ffi::PyLong_FromSsize_t(v as isize);
                if !object.is_null() && !key.is_null() {
                    ffi::PyDict_SetItem(env, key, object);
                }
                xdecref(key);
            }
            capi::release(std::mem::replace(&mut regs[v], null_mut()));
        }
        match failed {
            Some(i) => {
                ffi::PyErr_SetRaisedException(raised);
                Err(i)
            }
            None => Ok(()),
        }
    }

    /// Specializes the block again, right away, from what its profiling runs
    /// have seen so far: its versions are retired (the one that may be
    /// running stays alive), and the pipeline runs over the block anew.
    pub fn respecialize(&self, block: usize) {
        {
            let mut st = self.state.borrow_mut();
            let bs = &mut st.blocks[block];
            let old = std::mem::take(&mut bs.versions);
            bs.retired.extend(old);
            // One attempt for this, and one left so that this one may still
            // put guards on what was observed.
            bs.respec_left = bs.respec_left.max(2);
        }
        unsafe { self.specialize(block) };
    }

    /// `call` for the function that was handed to torch.vm, which
    /// is measured against the interpreter as its callees are.
    pub unsafe fn call_measured(&self, function: Obj, args: Obj, kwargs: Obj) -> Obj {
        // While it runs compiled, the frames the interpreter starts are
        // offered to it.
        let call = || {
            let _hooked = self.options.hook.then(|| Hooked::to(self));
            self.call(function, args, kwargs)
        };
        if !self.options.demote {
            return call();
        }
        ENTRY_CALLS.fetch_add(1, Ordering::Relaxed);
        let phase = self.settled.trial.get().phase;
        if phase == TRIAL_COMPILED {
            // The clock is read for a call only if something below the one
            // before it was being measured.
            let start = self.settled.watched.get().then(std::time::Instant::now);
            let events = TRIAL_EVENTS.load(Ordering::Relaxed);
            let measured = TRIAL_TIME.load(Ordering::Relaxed);
            let result = call();
            let watched = TRIAL_EVENTS.load(Ordering::Relaxed) != events;
            // Without the clock, it is not known how much of the call it was.
            let below = watched
                && start.is_none_or(|start| {
                    let measured = TRIAL_TIME.load(Ordering::Relaxed) - measured;
                    measured > start.elapsed().as_nanos() as u64 >> TRIAL_NOTICEABLE_SHIFT
                });
            self.settled.watched.set(watched);
            self.settled.below.set(below);
            return result;
        }
        if phase == TRIAL_DEMOTED {
            self.settled.below.set(false);
            let _nobody = Hooked::nobody();
            return ffi::PyObject_Call(function, args, kwargs);
        }
        // Not known before a call that is not itself part of a measurement.
        self.settled.watched.set(true);
        self.settled.below.set(true);
        self.trial_call(self, null_mut(), call, || ffi::PyObject_Call(function, args, kwargs))
    }

    /// Whether it is still open which of the two this function is run as,
    /// or was, during its last call, for calls below it that took a
    /// noticeable part of it: its calls are not yet as fast as they will be.
    pub fn is_measuring(&self) -> bool {
        let phase = self.settled.trial.get().phase;
        let open = phase == TRIAL_WARMUP || phase == TRIAL_SAMPLING;
        self.options.demote && (open || self.settled.below.get())
    }

    /// Whether this function was measured to run faster in the interpreter
    /// and is left to it.
    pub fn is_demoted(&self) -> bool {
        self.settled.trial.get().phase == TRIAL_DEMOTED
    }

    /// Calls a callee either compiled or through the interpreter, whichever
    /// the measurement of the two has shown to be faster; while that is
    /// still open, this call is part of the measurement.
    #[inline(always)]
    unsafe fn tiered(
        &self,
        compiled: *const Compiled,
        func: Obj,
        run_compiled: impl FnOnce() -> Obj,
        run_interpreted: impl FnOnce() -> Obj,
    ) -> Obj {
        let slot = &(*compiled).settled.trial;
        if slot.get().phase == TRIAL_COMPILED || !self.options.demote {
            return run_compiled();
        }
        self.trial_call(compiled, func, run_compiled, run_interpreted)
    }

    #[inline(never)]
    unsafe fn trial_call(
        &self,
        compiled: *const Compiled,
        func: Obj,
        run_compiled: impl FnOnce() -> Obj,
        run_interpreted: impl FnOnce() -> Obj,
    ) -> Obj {
        // What is left to the interpreter is left to it with all it calls:
        // that is what was measured.
        let run_interpreted = || {
            let _nobody = Hooked::nobody();
            run_interpreted()
        };
        if (*compiled).settled.trial.get().phase == TRIAL_DEMOTED {
            return run_interpreted();
        }
        // What is measured inside a call that is measured is part of it.
        let outermost = TRIAL_DEPTH.fetch_add(1, Ordering::Relaxed) == 0;
        let start = std::time::Instant::now();
        let result = self.trial_sample(compiled, func, run_compiled, run_interpreted);
        TRIAL_DEPTH.fetch_sub(1, Ordering::Relaxed);
        if outermost {
            TRIAL_TIME.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        result
    }

    /// Calls of a callee that was handed to the interpreter stop finding
    /// its compiled form.
    unsafe fn forget_compiled(&self, func: Obj) {
        if !func.is_null() {
            let code = ffi::PyFunction_GetCode(func);
            let globals = ffi::PyFunction_GetGlobals(func);
            let recent = self.callees.recent.get_unchecked((code as usize >> 4) & (RECENT_CALLEES - 1));
            recent.set((code as usize, globals as usize, DEMOTED));
        }
    }

    unsafe fn trial_sample(
        &self,
        compiled: *const Compiled,
        func: Obj,
        run_compiled: impl FnOnce() -> Obj,
        run_interpreted: impl FnOnce() -> Obj,
    ) -> Obj {
        let slot = &(*compiled).settled.trial;
        let mut trial = slot.get();
        if trial.phase == TRIAL_DEMOTED {
            return run_interpreted();
        }
        let epoch = PLAN_EPOCH.load(Ordering::Relaxed);
        if trial.epoch != epoch {
            trial = Trial { born: trial.born, ..Trial::start(epoch) };
        }
        trial.calls += 1;
        slot.set(trial);
        let events = TRIAL_EVENTS.fetch_add(1, Ordering::Relaxed) + 1;
        if trial.phase == TRIAL_WARMUP {
            let start = std::time::Instant::now();
            let result = run_compiled();
            let elapsed = start.elapsed().as_nanos() as u64;
            // The callee may have called itself: what is in the slot counts.
            let trial = slot.get();
            let warm = trial.calls
                >= trial_calls(elapsed, TRIAL_SLOW_WARMUP_CALLS, TRIAL_WARMUP_CALLS);
            if trial.phase == TRIAL_WARMUP && trial.epoch == epoch && warm {
                slot.set(Trial { phase: TRIAL_SAMPLING, ..trial });
            }
            return result;
        }
        let enough = trial.enough();
        // Which of the two a call measures is drawn, not alternated: calls
        // often come in a pattern (two layers of a model, one after the
        // other), and alternating would give each side one half of it.
        let as_compiled = if trial.interpreted_samples >= enough {
            true
        } else if trial.settled_samples >= enough {
            false
        } else {
            let mut state = TRIAL_DRAW.load(Ordering::Relaxed);
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            TRIAL_DRAW.store(state, Ordering::Relaxed);
            state & 1 == 0
        };
        // The outermost call that is measured starts with the devices idle
        // and ends when they are again, and so does any other that took
        // long enough to have given them work (see `TRIAL_SETTLE`).
        let outermost = TRIAL_DEPTH.load(Ordering::Relaxed) == 1;
        if outermost {
            trial_settle();
        }
        let start = std::time::Instant::now();
        let result = if as_compiled { run_compiled() } else { run_interpreted() };
        let mut elapsed = start.elapsed().as_nanos() as u64;
        if outermost || elapsed >= TRIAL_SETTLE_NS {
            trial_settle();
            elapsed = start.elapsed().as_nanos() as u64;
        }
        // The callee may have called itself: what is in the slot now counts.
        let mut trial = slot.get();
        if trial.phase != TRIAL_SAMPLING {
            return result;
        }
        // A call that raised is a sample like any other: a callee that
        // always raises (a lookup that fails, and is caught) would otherwise
        // never be decided, and nothing above it either.
        if !result.is_null() || !ffi::PyErr_Occurred().is_null() {
            if as_compiled {
                trial.compiled_best = trial.compiled_best.min(elapsed);
                trial.compiled_samples += 1;
                if TRIAL_EVENTS.load(Ordering::Relaxed) == events {
                    trial.settled_best = trial.settled_best.min(elapsed);
                    trial.settled_samples += 1;
                }
            } else {
                trial.interpreted_best = trial.interpreted_best.min(elapsed);
                trial.interpreted_samples += 1;
            }
        }
        let enough = trial.enough();
        let quarter = |t: u64| t + (t >> 2);
        let margin = trial.interpreted_best >> TRIAL_MARGIN_SHIFT;
        let seen = |samples: u8, enough: u8| samples >= enough && trial.interpreted_samples >= enough;
        let won = (seen(trial.compiled_samples, TRIAL_EARLY_SAMPLES)
            && quarter(trial.compiled_best) < trial.interpreted_best)
            || (seen(trial.compiled_samples, enough)
                && trial.interpreted_best + margin >= trial.compiled_best);
        let lost = (seen(trial.settled_samples, TRIAL_EARLY_SAMPLES)
            && quarter(trial.interpreted_best) < trial.settled_best)
            || (seen(trial.settled_samples, enough)
                && trial.interpreted_best + margin < trial.settled_best);
        if won {
            trial.phase = TRIAL_COMPILED;
        } else if lost {
            trial.phase = TRIAL_DEMOTED;
            self.forget_compiled(func);
        } else if trial.calls >= TRIAL_MAX_CALLS {
            // Still open: something below was compiled or measured during
            // every one of its calls.
            let age = (ENTRY_CALLS.load(Ordering::Relaxed) as u32).wrapping_sub(trial.born);
            let behind = seen(trial.compiled_samples, enough)
                && trial.interpreted_best + margin < trial.compiled_best;
            if age < TRIAL_YOUNG_ENTRIES {
                trial = Trial { born: trial.born, ..Trial::start(epoch) };
            } else if behind {
                // What it was measured to be is what it is: nothing says
                // that the calls below get any quieter.
                trial.phase = TRIAL_DEMOTED;
                self.forget_compiled(func);
            } else {
                trial.phase = TRIAL_COMPILED;
            }
        }
        slot.set(trial);
        result
    }

    /// Calls a compiled callee with arguments in the vectorcall layout.
    unsafe fn call_compiled(
        &self,
        compiled: &Compiled,
        func: Obj,
        receiver: Obj,
        args: &[Obj],
        kwnames: Obj,
    ) -> Obj {
        let nkw = if kwnames.is_null() { 0 } else { ffi::PyTuple_GET_SIZE(kwnames) as usize };
        let npos = args.len() - nkw;
        let positional = Args { receiver, items: args.as_ptr(), n: npos };
        if nkw == 0 {
            return compiled.enter(func, positional, null_mut());
        }
        let kwargs = ffi::PyDict_New();
        for i in 0..nkw {
            let name = ffi::PyTuple_GET_ITEM(kwnames, i as isize);
            if kwargs.is_null() || ffi::PyDict_SetItem(kwargs, name, args[npos + i]) < 0 {
                xdecref(kwargs);
                return null_mut();
            }
        }
        let result = compiled.enter(func, positional, kwargs);
        ffi::Py_DECREF(kwargs);
        result
    }

    /// The cell at index `i` of the function's closure (borrowed).
    unsafe fn closure_cell(&self, regs: &[Obj], i: u32) -> Result<Obj, ()> {
        let closure = ffi::PyFunction_GetClosure(*regs.get_unchecked(self.func.nvalues as usize));
        if closure.is_null() || ffi::PyTuple_GET_SIZE(closure) <= i as isize {
            set_error(ffi::PyExc_SystemError, "function has no such closure cell".to_string());
            return Err(());
        }
        Ok(ffi::PyTuple_GET_ITEM(closure, i as isize))
    }

    unsafe fn exec_op(&self, regs: &mut [Obj], op: &Op) -> OpResult {
        let f = &self.func;
        match op {
            Op::Const(c) => Ok(new_ref(f.consts.get_unchecked(*c as usize).as_ptr())),
            Op::LoadGlobal(name) => {
                let cache = self.global_cache.0.get_unchecked(*name as usize);
                let cached = cache.get();
                let versions =
                    (pyffi::dict_version(f.globals.as_ptr()), pyffi::dict_version(f.builtins.as_ptr()));
                if (cached.version, cached.builtins_version) == versions {
                    return Ok(new_ref(cached.value));
                }
                let name = f.consts[*name as usize].as_ptr();
                let mut v = ffi::PyDict_GetItemWithError(f.globals.as_ptr(), name);
                if v.is_null() {
                    if !ffi::PyErr_Occurred().is_null() {
                        return Err(());
                    }
                    v = ffi::PyDict_GetItemWithError(f.builtins.as_ptr(), name);
                    if v.is_null() {
                        if ffi::PyErr_Occurred().is_null() {
                            set_name_error(
                                ffi::PyExc_NameError,
                                format!("name '{}' is not defined", pyffi::str_to_string(name)),
                                name,
                            );
                        }
                        return Err(());
                    }
                }
                cache.set(LoadCache {
                    owner: null_mut(),
                    version: versions.0,
                    builtins_version: versions.1,
                    value: v,
                });
                Ok(new_ref(v))
            }
            Op::LoadDeref(i) => {
                let cell = self.closure_cell(regs, *i)?;
                let v = pyffi::PyCell_Get(cell);
                if v.is_null() {
                    if ffi::PyErr_Occurred().is_null() {
                        let name = f.freevars.get(*i as usize).map_or("?", |s| s.as_str());
                        let name_obj = ffi::PyUnicode_FromStringAndSize(name.as_ptr().cast(), name.len() as isize);
                        set_name_error(
                            ffi::PyExc_NameError,
                            format!(
                                "cannot access free variable '{}' where it is not associated \
                                 with a value in enclosing scope",
                                name
                            ),
                            name_obj,
                        );
                        xdecref(name_obj);
                    }
                    return Err(());
                }
                Ok(v)
            }
            Op::ImportName { name, fromlist, level } => {
                // What the interpreter does: builtins.__import__(name, globals,
                // locals, fromlist, level).
                let import = ffi::PyDict_GetItemString(f.builtins.as_ptr(), c"__import__".as_ptr());
                if import.is_null() {
                    set_error(ffi::PyExc_ImportError, "__import__ not found".to_string());
                    return Err(());
                }
                let args = [
                    f.consts[*name as usize].as_ptr(),
                    f.globals.as_ptr(),
                    ffi::Py_None(),
                    self.obj(regs, *fromlist),
                    self.obj(regs, *level),
                ];
                // The import system, and the module it runs, are the
                // interpreter's.
                let _nobody = Hooked::nobody();
                checked(ffi::PyObject_Vectorcall(import, args.as_ptr(), 5, null_mut()))
            }
            Op::ImportFrom(module, name) => {
                let module = self.obj(regs, *module);
                let name = f.consts[*name as usize].as_ptr();
                let v = ffi::PyObject_GetAttr(module, name);
                if !v.is_null() || ffi::PyErr_ExceptionMatches(ffi::PyExc_AttributeError) == 0 {
                    return checked(v);
                }
                // A submodule that is still being imported is in sys.modules
                // before it is an attribute of its package.
                ffi::PyErr_Clear();
                let package = ffi::PyObject_GetAttrString(module, c"__name__".as_ptr());
                let mut found = null_mut();
                if !package.is_null() {
                    let full = ffi::PyUnicode_FromFormat(c"%U.%U".as_ptr(), package, name);
                    if !full.is_null() {
                        found = pyffi::PyImport_GetModule(full);
                        ffi::Py_DECREF(full);
                    }
                }
                if found.is_null() {
                    ffi::PyErr_Clear();
                    set_error(
                        ffi::PyExc_ImportError,
                        format!(
                            "cannot import name '{}' from '{}'",
                            pyffi::str_to_string(name),
                            if package.is_null() { "?".to_string() } else { pyffi::str_to_string(package) }
                        ),
                    );
                }
                xdecref(package);
                checked(found)
            }
            Op::Locals(callee, entries) => {
                let callee = self.obj(regs, *callee);
                let (locals, vars, dir) = frame_readers();
                if callee != locals && callee != vars && callee != dir {
                    return checked(ffi::PyObject_CallNoArgs(callee));
                }
                let dict = ffi::PyDict_New();
                if dict.is_null() {
                    return Err(());
                }
                for (name, v, is_cell) in entries {
                    if reg(regs, *v).is_null() {
                        continue;
                    }
                    let held = self.obj(regs, *v);
                    let value = if *is_cell && ffi::Py_TYPE(held) == std::ptr::addr_of_mut!(pyffi::PyCell_Type) {
                        pyffi::PyCell_Get(held)
                    } else {
                        new_ref(held)
                    };
                    if value.is_null() {
                        // A cell without a value: a variable that has none.
                        ffi::PyErr_Clear();
                        continue;
                    }
                    let failed = ffi::PyDict_SetItem(dict, f.consts[*name as usize].as_ptr(), value) < 0;
                    ffi::Py_DECREF(value);
                    if failed {
                        ffi::Py_DECREF(dict);
                        return Err(());
                    }
                }
                if callee != dir {
                    return Ok(dict);
                }
                let names = ffi::PyDict_Keys(dict);
                ffi::Py_DECREF(dict);
                if names.is_null() || ffi::PyList_Sort(names) < 0 {
                    xdecref(names);
                    return Err(());
                }
                Ok(names)
            }
            // The block loop runs the last three; it has the frames.
            Op::Hold(_) | Op::FrameEnter(_) | Op::FrameLeave | Op::StoreLocal(..) => Ok(null_mut()),
            Op::CaughtLasti => Ok(new_ref(ffi::Py_None())),
            Op::LoadSlot(slot) => {
                let o = reg(regs, *slot);
                capi::retain(o);
                Ok(o)
            }
            Op::StoreSlot(slot, v) => {
                let o = reg(regs, *v);
                capi::retain(o);
                capi::release(std::mem::replace(regs.get_unchecked_mut(*slot as usize), o));
                Ok(null_mut())
            }
            Op::CaughtException => Ok(take(regs, f.exc_reg)),
            Op::PushExcInfo(exc) => {
                let previous = pyffi::PyErr_GetHandledException();
                pyffi::PyErr_SetHandledException(self.obj(regs, *exc));
                Ok(if previous.is_null() { new_ref(ffi::Py_None()) } else { previous })
            }
            Op::PopExcInfo(previous) => {
                pyffi::PyErr_SetHandledException(self.obj(regs, *previous));
                Ok(null_mut())
            }
            Op::ExcMatch(exc, class) => {
                let class = self.obj(regs, *class);
                let is_class = |c: Obj| ffi::PyExceptionClass_Check(c) != 0;
                let valid = if ffi::PyTuple_Check(class) != 0 {
                    (0..ffi::PyTuple_GET_SIZE(class)).all(|i| is_class(ffi::PyTuple_GET_ITEM(class, i)))
                } else {
                    is_class(class)
                };
                if !valid {
                    set_error(
                        ffi::PyExc_TypeError,
                        "catching classes that do not inherit from BaseException is not allowed"
                            .to_string(),
                    );
                    return Err(());
                }
                Ok(py_bool(ffi::PyErr_GivenExceptionMatches(self.obj(regs, *exc), class) != 0))
            }
            Op::LoadSpecial(obj, name) => {
                let o = self.obj(regs, *obj);
                let ty = ffi::Py_TYPE(o);
                let attr = pyffi::_PyType_Lookup(ty, f.consts[*name as usize].as_ptr());
                if attr.is_null() {
                    let name = pyffi::str_to_string(f.consts[*name as usize].as_ptr());
                    set_error(ffi::PyExc_TypeError, not_a_context_manager(o, &name));
                    return Err(());
                }
                match (*ffi::Py_TYPE(attr)).tp_descr_get {
                    Some(bind) => checked(bind(attr, o, ty as Obj)),
                    None => Ok(new_ref(attr)),
                }
            }
            Op::WithExceptStart { exit, exc } => {
                let exit = self.obj(regs, *exit);
                let exc = self.obj(regs, *exc);
                let traceback = ffi::PyException_GetTraceback(exc);
                let args = [
                    ffi::Py_TYPE(exc) as Obj,
                    exc,
                    if traceback.is_null() { ffi::Py_None() } else { traceback },
                ];
                let r = ffi::PyObject_Vectorcall(exit, args.as_ptr(), 3, null_mut());
                xdecref(traceback);
                checked(r)
            }
            Op::StoreDeref(i, v) => {
                let cell = self.closure_cell(regs, *i)?;
                let value = if *v == UNDEF { null_mut() } else { self.obj(regs, *v) };
                if pyffi::PyCell_Set(cell, value) < 0 {
                    return Err(());
                }
                Ok(null_mut())
            }
            Op::LoadClosure(i) => Ok(new_ref(self.closure_cell(regs, *i)?)),
            Op::MakeCell(v) => {
                let initial = if *v == UNDEF { null_mut() } else { self.obj(regs, *v) };
                checked(pyffi::PyCell_New(initial))
            }
            Op::CellGet(cell, name) => {
                let v = pyffi::PyCell_Get(self.obj(regs, *cell));
                if v.is_null() {
                    if ffi::PyErr_Occurred().is_null() {
                        set_error(
                            ffi::PyExc_UnboundLocalError,
                            format!(
                                "cannot access local variable '{}' where it is not associated \
                                 with a value",
                                pyffi::str_to_string(f.consts[*name as usize].as_ptr())
                            ),
                        );
                    }
                    return Err(());
                }
                Ok(v)
            }
            Op::CellSet(cell, v) => {
                let value = if *v == UNDEF { null_mut() } else { self.obj(regs, *v) };
                if pyffi::PyCell_Set(self.obj(regs, *cell), value) < 0 {
                    return Err(());
                }
                Ok(null_mut())
            }
            Op::MakeFunction { code, defaults, kwdefaults, annotations, annotate, closure } => {
                let func = ffi::PyFunction_New(self.obj(regs, *code), f.globals.as_ptr());
                if func.is_null() {
                    return Err(());
                }
                let mut ok = true;
                if *closure != NONE {
                    ok &= ffi::PyFunction_SetClosure(func, self.obj(regs, *closure)) == 0;
                }
                if *defaults != NONE {
                    ok &= ffi::PyFunction_SetDefaults(func, self.obj(regs, *defaults)) == 0;
                }
                if *kwdefaults != NONE {
                    ok &= ffi::PyFunction_SetKwDefaults(func, self.obj(regs, *kwdefaults)) == 0;
                }
                if *annotations != NONE {
                    // The bytecode passes a tuple of names and values.
                    let pairs = self.obj(regs, *annotations);
                    let dict = ffi::PyDict_New();
                    ok &= !dict.is_null();
                    let n = if ok && ffi::PyTuple_Check(pairs) != 0 { ffi::PyTuple_GET_SIZE(pairs) } else { 0 };
                    let mut i = 0;
                    while ok && i + 1 < n {
                        ok &= ffi::PyDict_SetItem(
                            dict,
                            ffi::PyTuple_GET_ITEM(pairs, i),
                            ffi::PyTuple_GET_ITEM(pairs, i + 1),
                        ) == 0;
                        i += 2;
                    }
                    ok = ok && ffi::PyFunction_SetAnnotations(func, dict) == 0;
                    xdecref(dict);
                }
                if *annotate != NONE {
                    // As SET_FUNCTION_ATTRIBUTE does it, down to the name
                    // the function that computes them goes by.
                    let annotate = self.obj(regs, *annotate);
                    ok = ok && ffi::PyObject_SetAttrString(func, c"__annotate__".as_ptr(), annotate) == 0;
                    if ok && ffi::PyFunction_Check(annotate) != 0 {
                        let of = ffi::PyObject_GetAttrString(func, c"__qualname__".as_ptr());
                        let name = if of.is_null() {
                            null_mut()
                        } else {
                            ffi::PyUnicode_FromFormat(c"%U.__annotate__".as_ptr(), of)
                        };
                        ok = !name.is_null()
                            && ffi::PyObject_SetAttrString(annotate, c"__qualname__".as_ptr(), name) == 0;
                        xdecref(of);
                        xdecref(name);
                    }
                }
                if !ok {
                    ffi::Py_DECREF(func);
                    return Err(());
                }
                Ok(func)
            }
            Op::LoadAttr(a, name) => {
                let obj = self.obj(regs, *a);
                // torch.add and friends: an attribute of a plain module is an
                // entry of its dict.
                if ffi::PyModule_CheckExact(obj) != 0 && *self.plain_name.get_unchecked(*name as usize)
                {
                    let dict = ffi::PyModule_GetDict(obj);
                    if !dict.is_null() {
                        let cache = self.module_attr_cache.0.get_unchecked(*name as usize);
                        let cached = cache.get();
                        let version = pyffi::dict_version(dict);
                        if cached.owner == obj && cached.version == version {
                            return Ok(new_ref(cached.value));
                        }
                        let v = ffi::PyDict_GetItemWithError(dict, f.consts[*name as usize].as_ptr());
                        if !v.is_null() {
                            cache.set(LoadCache { owner: obj, version, builtins_version: 0, value: v });
                            return Ok(new_ref(v));
                        }
                        if !ffi::PyErr_Occurred().is_null() {
                            return Err(());
                        }
                    }
                }
                let name = f.consts[*name as usize].as_ptr();
                if self.options.shortcuts {
                    let found = crate::shortcuts::load_attr(obj, name);
                    if !found.is_null() {
                        return Ok(found);
                    }
                    if !ffi::PyErr_Occurred().is_null() {
                        return Err(());
                    }
                }
                checked(ffi::PyObject_GetAttr(obj, name))
            }
            Op::StoreAttr { obj, name, val } => status(ffi::PyObject_SetAttr(
                self.obj(regs, *obj),
                f.consts[*name as usize].as_ptr(),
                if *val == UNDEF { null_mut() } else { self.obj(regs, *val) },
            )),
            Op::StoreGlobal(name, v) if *v == UNDEF => {
                let name = f.consts[*name as usize].as_ptr();
                if ffi::PyDict_DelItem(f.globals.as_ptr(), name) < 0 {
                    if ffi::PyErr_ExceptionMatches(ffi::PyExc_KeyError) != 0 {
                        ffi::PyErr_Clear();
                        set_name_error(
                            ffi::PyExc_NameError,
                            format!("name '{}' is not defined", pyffi::str_to_string(name)),
                            name,
                        );
                    }
                    return Err(());
                }
                Ok(null_mut())
            }
            Op::StoreGlobal(name, v) => status(ffi::PyDict_SetItem(
                f.globals.as_ptr(),
                f.consts[*name as usize].as_ptr(),
                self.obj(regs, *v),
            )),
            Op::Binary(kind, a, b) => checked(binary(*kind, self.obj(regs, *a), self.obj(regs, *b))),
            Op::Unary(kind, a) => {
                let a = self.obj(regs, *a);
                match kind {
                    UnaryKind::Neg => checked(ffi::PyNumber_Negative(a)),
                    UnaryKind::Pos => checked(ffi::PyNumber_Positive(a)),
                    UnaryKind::Invert => checked(ffi::PyNumber_Invert(a)),
                    UnaryKind::Not => {
                        let r = ffi::PyObject_IsTrue(a);
                        if r < 0 {
                            return Err(());
                        }
                        Ok(py_bool(r == 0))
                    }
                }
            }
            Op::Compare(kind, a, b) => {
                checked(ffi::PyObject_RichCompare(self.obj(regs, *a), self.obj(regs, *b), *kind as i32))
            }
            Op::Is(a, b, invert) => Ok(py_bool((self.obj(regs, *a) == self.obj(regs, *b)) != *invert)),
            Op::Contains(item, container, invert) => {
                let r = ffi::PySequence_Contains(self.obj(regs, *container), self.obj(regs, *item));
                if r < 0 {
                    return Err(());
                }
                Ok(py_bool((r == 1) != *invert))
            }
            Op::ToBool(a) => {
                let a = self.obj(regs, *a);
                if a == ffi::Py_True() || a == ffi::Py_False() {
                    return Ok(new_ref(a));
                }
                let r = ffi::PyObject_IsTrue(a);
                if r < 0 {
                    return Err(());
                }
                Ok(py_bool(r != 0))
            }
            // An unboxed tensor is not None; no need to box it to find out.
            Op::IsNone(a, invert) => Ok(py_bool((reg(regs, *a) == ffi::Py_None()) != *invert)),
            Op::GetItem(a, k) => checked(ffi::PyObject_GetItem(self.obj(regs, *a), self.obj(regs, *k))),
            Op::SetItem { obj, key, val } => {
                status(ffi::PyObject_SetItem(self.obj(regs, *obj), self.obj(regs, *key), self.obj(regs, *val)))
            }
            Op::DelItem(a, k) => status(ffi::PyObject_DelItem(self.obj(regs, *a), self.obj(regs, *k))),
            Op::BuildTuple(vs) => {
                let t = ffi::PyTuple_New(vs.len() as isize);
                if t.is_null() {
                    return Err(());
                }
                for (i, v) in vs.iter().enumerate() {
                    ffi::PyTuple_SET_ITEM(t, i as isize, new_ref(self.obj(regs, *v)));
                }
                Ok(t)
            }
            Op::BuildList(vs) => {
                let l = ffi::PyList_New(vs.len() as isize);
                if l.is_null() {
                    return Err(());
                }
                for (i, v) in vs.iter().enumerate() {
                    ffi::PyList_SET_ITEM(l, i as isize, new_ref(self.obj(regs, *v)));
                }
                Ok(l)
            }
            Op::BuildSet(vs) => {
                let s = ffi::PySet_New(null_mut());
                if s.is_null() {
                    return Err(());
                }
                for v in vs {
                    if ffi::PySet_Add(s, self.obj(regs, *v)) < 0 {
                        ffi::Py_DECREF(s);
                        return Err(());
                    }
                }
                Ok(s)
            }
            Op::BuildMap(vs) => {
                let d = ffi::PyDict_New();
                if d.is_null() {
                    return Err(());
                }
                for kv in vs.chunks(2) {
                    if ffi::PyDict_SetItem(d, self.obj(regs, kv[0]), self.obj(regs, kv[1])) < 0 {
                        ffi::Py_DECREF(d);
                        return Err(());
                    }
                }
                Ok(d)
            }
            Op::BuildSlice(a, b, c) => {
                let step = if *c == NONE { null_mut() } else { self.obj(regs, *c) };
                checked(ffi::PySlice_New(self.obj(regs, *a), self.obj(regs, *b), step))
            }
            Op::BuildString(vs) => {
                let t = ffi::PyTuple_New(vs.len() as isize);
                if t.is_null() {
                    return Err(());
                }
                for (i, v) in vs.iter().enumerate() {
                    ffi::PyTuple_SET_ITEM(t, i as isize, new_ref(self.obj(regs, *v)));
                }
                let empty = ffi::PyUnicode_New(0, 0);
                let s = if empty.is_null() { null_mut() } else { ffi::PyUnicode_Join(empty, t) };
                xdecref(empty);
                ffi::Py_DECREF(t);
                checked(s)
            }
            Op::Format { val, conv, spec } => {
                let val = self.obj(regs, *val);
                let converted = match conv & 3 {
                    1 => ffi::PyObject_Str(val),
                    2 => ffi::PyObject_Repr(val),
                    3 => ffi::PyObject_ASCII(val),
                    _ => new_ref(val),
                };
                if converted.is_null() {
                    return Err(());
                }
                if conv & CONVERT_ONLY != 0 || (*spec == NONE && ffi::PyUnicode_CheckExact(converted) != 0) {
                    return Ok(converted);
                }
                let spec = if *spec == NONE { null_mut() } else { self.obj(regs, *spec) };
                let s = ffi::PyObject_Format(converted, spec);
                ffi::Py_DECREF(converted);
                checked(s)
            }
            Op::ListAppend(l, v) => status(ffi::PyList_Append(self.obj(regs, *l), self.obj(regs, *v))),
            Op::SetAdd(s, v) => status(ffi::PySet_Add(self.obj(regs, *s), self.obj(regs, *v))),
            Op::MapAdd(d, k, v) => {
                status(ffi::PyDict_SetItem(self.obj(regs, *d), self.obj(regs, *k), self.obj(regs, *v)))
            }
            Op::ListExtend(l, it) => {
                let it = self.obj(regs, *it);
                let r = ffi::PySequence_InPlaceConcat(self.obj(regs, *l), it);
                if r.is_null() {
                    if ffi::PyErr_ExceptionMatches(ffi::PyExc_TypeError) != 0 && !is_iterable(it) {
                        ffi::PyErr_Clear();
                        set_error(
                            ffi::PyExc_TypeError,
                            format!("Value after * must be an iterable, not {}", pyffi::type_name(it)),
                        );
                    }
                    return Err(());
                }
                ffi::Py_DECREF(r);
                Ok(null_mut())
            }
            Op::SetUpdate(s, it) => {
                let name = ffi::PyUnicode_InternFromString(c"update".as_ptr());
                if name.is_null() {
                    return Err(());
                }
                let args = [self.obj(regs, *s), self.obj(regs, *it)];
                let r = ffi::PyObject_VectorcallMethod(name, args.as_ptr(), 2, null_mut());
                ffi::Py_DECREF(name);
                if r.is_null() {
                    return Err(());
                }
                ffi::Py_DECREF(r);
                Ok(null_mut())
            }
            Op::DictUpdate(d, o) => {
                let o = self.obj(regs, *o);
                if ffi::PyDict_Update(self.obj(regs, *d), o) < 0 {
                    if ffi::PyErr_ExceptionMatches(ffi::PyExc_AttributeError) != 0 {
                        ffi::PyErr_Clear();
                        set_error(
                            ffi::PyExc_TypeError,
                            format!("'{}' object is not a mapping", pyffi::type_name(o)),
                        );
                    }
                    return Err(());
                }
                Ok(null_mut())
            }
            Op::DictMerge(d, o, func) => {
                let (d, o) = (self.obj(regs, *d), self.obj(regs, *o));
                if pyffi::_PyDict_MergeEx(d, o, 2) < 0 {
                    // format_kwargs_error of ceval.c.
                    if ffi::PyErr_ExceptionMatches(ffi::PyExc_AttributeError) != 0 {
                        ffi::PyErr_Clear();
                        set_error(
                            ffi::PyExc_TypeError,
                            format!(
                                "{} argument after ** must be a mapping, not {}",
                                pyffi::function_str(self.obj(regs, *func)),
                                pyffi::type_name(o)
                            ),
                        );
                    } else if ffi::PyErr_ExceptionMatches(ffi::PyExc_KeyError) != 0 {
                        let exc = ffi::PyErr_GetRaisedException();
                        let args = ffi::PyObject_GetAttrString(exc, c"args".as_ptr());
                        if !args.is_null() && ffi::PyTuple_Check(args) != 0 && ffi::PyTuple_GET_SIZE(args) == 1 {
                            let key = ffi::PyObject_Str(ffi::PyTuple_GET_ITEM(args, 0));
                            let key_text = if key.is_null() { "?".to_string() } else { pyffi::str_to_string(key) };
                            xdecref(key);
                            ffi::PyErr_Clear();
                            set_error(
                                ffi::PyExc_TypeError,
                                format!(
                                    "{} got multiple values for keyword argument '{}'",
                                    pyffi::function_str(self.obj(regs, *func)),
                                    key_text
                                ),
                            );
                            ffi::Py_DECREF(exc);
                        } else {
                            ffi::PyErr_Clear();
                            ffi::PyErr_SetRaisedException(exc);
                        }
                        xdecref(args);
                    }
                    return Err(());
                }
                Ok(null_mut())
            }
            Op::ListToTuple(l) => checked(ffi::PyList_AsTuple(self.obj(regs, *l))),
            Op::SuperObject { global_super, class, this, explicit } => {
                let callee = self.obj(regs, *global_super);
                let args = [self.obj(regs, *class), self.obj(regs, *this)];
                let builtin = callee == std::ptr::addr_of_mut!(ffi::PySuper_Type) as Obj;
                let nargs = if *explicit || builtin { 2 } else { 0 };
                checked(ffi::PyObject_Vectorcall(callee, args.as_ptr(), nargs, null_mut()))
            }
            Op::CallObject(callee, args) => {
                let mut small: [Obj; 8] = [null_mut(); 8];
                let mut big: Vec<Obj>;
                let buf: &mut [Obj] = if args.len() <= small.len() {
                    &mut small[..args.len()]
                } else {
                    big = vec![null_mut(); args.len()];
                    &mut big[..]
                };
                for (i, v) in args.iter().enumerate() {
                    buf[i] = self.obj(regs, *v);
                }
                checked(ffi::PyObject_Vectorcall(callee.as_ptr(), buf.as_ptr(), args.len(), null_mut()))
            }
            Op::GetLen(v) => {
                let len = ffi::PyObject_Length(self.obj(regs, *v));
                if len < 0 {
                    return Err(());
                }
                checked(ffi::PyLong_FromSsize_t(len))
            }
            Op::MatchKind(v, mapping) => {
                let flag =
                    if *mapping { ffi::Py_TPFLAGS_MAPPING } else { ffi::Py_TPFLAGS_SEQUENCE };
                let holds = ffi::PyType_HasFeature(ffi::Py_TYPE(self.obj(regs, *v)), flag) != 0;
                Ok(new_ref(if holds { ffi::Py_True() } else { ffi::Py_False() }))
            }
            Op::MatchKeys(subject, keys) => {
                checked(match_keys(self.obj(regs, *subject), self.obj(regs, *keys)))
            }
            Op::MatchClass { subject, class, nargs, names } => checked(match_class(
                self.obj(regs, *subject),
                self.obj(regs, *class),
                *nargs as isize,
                self.obj(regs, *names),
            )),
            Op::UnpackEx(seq, before, after) => {
                let seq = self.obj(regs, *seq);
                let items = ffi::PySequence_List(seq);
                if items.is_null() {
                    if ffi::PyErr_ExceptionMatches(ffi::PyExc_TypeError) != 0 {
                        ffi::PyErr_Clear();
                        set_error(
                            ffi::PyExc_TypeError,
                            format!("cannot unpack non-iterable {} object", pyffi::type_name(seq)),
                        );
                    }
                    return Err(());
                }
                let (before, after) = (*before as isize, *after as isize);
                let size = ffi::PyList_GET_SIZE(items);
                if size < before + after {
                    ffi::Py_DECREF(items);
                    set_error(
                        ffi::PyExc_ValueError,
                        format!(
                            "not enough values to unpack (expected at least {}, got {})",
                            before + after,
                            size
                        ),
                    );
                    return Err(());
                }
                let out = ffi::PyTuple_New(before + 1 + after);
                let middle = ffi::PyList_GetSlice(items, before, size - after);
                if out.is_null() || middle.is_null() {
                    ffi::Py_DECREF(items);
                    xdecref(out);
                    xdecref(middle);
                    return Err(());
                }
                for i in 0..before {
                    ffi::PyTuple_SET_ITEM(out, i, new_ref(ffi::PyList_GET_ITEM(items, i)));
                }
                ffi::PyTuple_SET_ITEM(out, before, middle);
                for i in 0..after {
                    let item = ffi::PyList_GET_ITEM(items, size - after + i);
                    ffi::PyTuple_SET_ITEM(out, before + 1 + i, new_ref(item));
                }
                ffi::Py_DECREF(items);
                Ok(out)
            }
            Op::Unpack(seq, n) => {
                let seq = self.obj(regs, *seq);
                let n = *n as isize;
                let t = if ffi::PyTuple_CheckExact(seq) != 0 {
                    new_ref(seq)
                } else {
                    let t = ffi::PySequence_Tuple(seq);
                    if t.is_null() {
                        if ffi::PyErr_ExceptionMatches(ffi::PyExc_TypeError) != 0 {
                            ffi::PyErr_Clear();
                            set_error(
                                ffi::PyExc_TypeError,
                                format!("cannot unpack non-iterable {} object", pyffi::type_name(seq)),
                            );
                        }
                        return Err(());
                    }
                    t
                };
                let size = ffi::PyTuple_GET_SIZE(t);
                if size != n {
                    ffi::Py_DECREF(t);
                    // From 3.14 on the interpreter says how many there are
                    // of what it knows the length of.
                    let sized = cfg!(Py_3_14)
                        && (ffi::PyList_CheckExact(seq) != 0
                            || ffi::PyTuple_CheckExact(seq) != 0
                            || ffi::PyDict_CheckExact(seq) != 0);
                    set_error(
                        ffi::PyExc_ValueError,
                        if size > n && sized {
                            format!("too many values to unpack (expected {}, got {})", n, size)
                        } else if size > n {
                            format!("too many values to unpack (expected {})", n)
                        } else {
                            format!("not enough values to unpack (expected {}, got {})", n, size)
                        },
                    );
                    return Err(());
                }
                Ok(t)
            }
            Op::TupleGet(t, i) => Ok(new_ref(ffi::PyTuple_GET_ITEM(self.obj(regs, *t), *i as isize))),
            Op::Object(o) => Ok(new_ref(o.as_ptr())),
            Op::Part(part, v) => {
                let held = self.obj(regs, *v);
                let function = ffi::PyFunction_Check(held) != 0;
                let item = |tuple: Obj, i: u32| {
                    let there = !tuple.is_null() && ffi::PyTuple_GET_SIZE(tuple) > i as isize;
                    if there { ffi::PyTuple_GET_ITEM(tuple, i as isize) } else { null_mut() }
                };
                let found = match part {
                    Part::Cell(i) if function => {
                        let cell = item(ffi::PyFunction_GetClosure(held), *i);
                        if cell.is_null() { null_mut() } else { pyffi::PyCell_Get(cell) }
                    }
                    Part::Default(i) if function => {
                        let default = item(ffi::PyFunction_GetDefaults(held), *i);
                        ffi::Py_XINCREF(default);
                        default
                    }
                    Part::KwDefault(name) if function => {
                        let defaults = ffi::PyFunction_GetKwDefaults(held);
                        let default =
                            if defaults.is_null() { null_mut() } else { ffi::PyDict_GetItemWithError(defaults, name.as_ptr()) };
                        ffi::Py_XINCREF(default);
                        default
                    }
                    Part::Receiver if ffi::Py_TYPE(held) == std::ptr::addr_of_mut!(pyffi::PyMethod_Type) => {
                        new_ref(pyffi::PyMethod_Self(held))
                    }
                    Part::Target => self.target(held).map_or(null_mut(), |(func, ..)| new_ref(func)),
                    Part::Type => new_ref(ffi::Py_TYPE(held) as Obj),
                    Part::OfType(name) => {
                        let found = pyffi::_PyType_Lookup(ffi::Py_TYPE(held), name.as_ptr());
                        ffi::Py_XINCREF(found);
                        found
                    }
                    Part::Descriptor(name) => {
                        let found = graph::data_descriptor(held, name.as_ptr());
                        ffi::Py_XINCREF(found);
                        found
                    }
                    Part::Getter if ffi::Py_TYPE(held) == std::ptr::addr_of_mut!(ffi::PyProperty_Type) => {
                        ffi::PyObject_GetAttrString(held, c"fget".as_ptr())
                    }
                    Part::Index(i) => {
                        let plain = ffi::PyTuple_CheckExact(held) != 0 || ffi::PyList_CheckExact(held) != 0;
                        let there = plain && ffi::PyObject_Size(held) > *i as isize;
                        let found = if there { graph::sequence_item(held, *i as isize) } else { null_mut() };
                        ffi::Py_XINCREF(found);
                        found
                    }
                    Part::Entry(key) if ffi::PyDict_CheckExact(held) != 0 => {
                        let found = ffi::PyDict_GetItemWithError(held, key.as_ptr());
                        ffi::Py_XINCREF(found);
                        found
                    }
                    _ => null_mut(),
                };
                if found.is_null() && ffi::PyErr_Occurred().is_null() {
                    set_error(ffi::PyExc_SystemError, "a function is not what a path took it for".to_string());
                }
                checked(found)
            }
            Op::Call { callee, args, kwnames } => {
                let callee = self.obj(regs, *callee);
                checked(self.vectorcall(regs, callee, args, *kwnames))
            }
            Op::CallMethod { recv, name, args, kwnames } => {
                // Looks the method up on the type and calls it with the
                // receiver, without materializing a bound method object.
                let mut small: [Obj; 9] = [null_mut(); 9];
                let mut big: Vec<Obj>;
                let buf: &mut [Obj] = if args.len() < small.len() {
                    &mut small[..]
                } else {
                    big = vec![null_mut(); args.len() + 1];
                    &mut big[..]
                };
                buf[0] = self.obj(regs, *recv);
                for (i, v) in args.iter().enumerate() {
                    buf[i + 1] = self.obj(regs, *v);
                }
                let (kw, nkw) = if *kwnames == NONE {
                    (null_mut(), 0)
                } else {
                    let kw = f.consts[*kwnames as usize].as_ptr();
                    (kw, ffi::PyTuple_GET_SIZE(kw) as usize)
                };
                let mut method: Obj = null_mut();
                let attr = f.consts[*name as usize].as_ptr();
                if self.options.shortcuts {
                    method = crate::shortcuts::load_attr(buf[0], attr);
                    if method.is_null() && !ffi::PyErr_Occurred().is_null() {
                        return Err(());
                    }
                }
                // Found through a shortcut, it is the attribute, bound.
                let unbound = if method.is_null() {
                    pyffi::_PyObject_GetMethod(buf[0], attr, &mut method)
                } else {
                    0
                };
                if method.is_null() {
                    return Err(());
                }
                // A method takes the receiver as its first argument; any
                // other attribute is called on its own.
                let (first, receiver) = if unbound == 1 { (0, buf[0]) } else { (1, null_mut()) };
                if unbound == 1 && self.callees.callee_asked.get() {
                    self.callees.callee_asked.set(false);
                    if ffi::PyFunction_Check(method) != 0 {
                        self.note_callee(method, CalleeKind::Method);
                    }
                }
                // A function found on the type runs compiled, bound to the
                // receiver; anything else that was found is called as any
                // callee is, which finds what it comes down to.
                let compiled = if !self.options.callees {
                    None
                } else if unbound == 1 {
                    (ffi::PyFunction_Check(method) != 0)
                        .then(|| self.compiled_function(method))
                        .flatten()
                        .map(|compiled| (compiled, method, receiver))
                } else {
                    self.compiled_callee(method)
                };
                let interpreted = || {
                    ffi::PyObject_Vectorcall(
                        method,
                        buf.as_ptr().add(first),
                        args.len() + 1 - first - nkw,
                        kw,
                    )
                };
                let r = match compiled {
                    Some((compiled, func, bound_to)) => self.tiered(
                        compiled,
                        func,
                        || self.call_compiled(&*compiled, func, bound_to, &buf[1..args.len() + 1], kw),
                        interpreted,
                    ),
                    None => interpreted(),
                };
                ffi::Py_DECREF(method);
                checked(r)
            }
            Op::CallEx { callee, args, kwargs } => {
                let args = self.obj(regs, *args);
                let tuple = if ffi::PyTuple_CheckExact(args) != 0 {
                    new_ref(args)
                } else {
                    if !is_iterable(args) {
                        set_error(
                            ffi::PyExc_TypeError,
                            format!(
                                "{} argument after * must be an iterable, not {}",
                                pyffi::function_str(self.obj(regs, *callee)),
                                pyffi::type_name(args)
                            ),
                        );
                        return Err(());
                    }
                    let t = ffi::PySequence_Tuple(args);
                    if t.is_null() {
                        return Err(());
                    }
                    t
                };
                let kwargs = if *kwargs == NONE { null_mut() } else { self.obj(regs, *kwargs) };
                let callee = self.obj(regs, *callee);
                let plain_kwargs = kwargs.is_null() || ffi::PyDict_CheckExact(kwargs) != 0;
                let interpreted = || ffi::PyObject_Call(callee, tuple, kwargs);
                let r = match self.compiled_callee(callee).filter(|_| plain_kwargs) {
                    Some((compiled, func, receiver)) => self.tiered(
                        compiled,
                        func,
                        || (*compiled).enter(func, Args::tuple(receiver, tuple), kwargs),
                        interpreted,
                    ),
                    None => interpreted(),
                };
                ffi::Py_DECREF(tuple);
                checked(r)
            }
            Op::GetIter(a) => checked(ffi::PyObject_GetIter(self.obj(regs, *a))),
            Op::IterNext(it) => {
                let v = ffi::PyIter_Next(self.obj(regs, *it));
                if v.is_null() && !ffi::PyErr_Occurred().is_null() {
                    return Err(());
                }
                Ok(v)
            }
            Op::IsFunction { v, func, code } => {
                let o = self.obj(regs, *v);
                // An inlined function is not one the interpreter enters.
                Ok(py_bool(
                    o == f.consts.get_unchecked(*func as usize).as_ptr()
                        && ffi::PyFunction_GetCode(o)
                            == f.consts.get_unchecked(*code as usize).as_ptr()
                        && !capi::traced(),
                ))
            }
            Op::CheckBound(v, name) => {
                if reg(regs, *v).is_null() {
                    set_error(
                        ffi::PyExc_UnboundLocalError,
                        format!(
                            "cannot access local variable '{}' where it is not associated with a \
                             value",
                            pyffi::str_to_string(f.consts[*name as usize].as_ptr())
                        ),
                    );
                    return Err(());
                }
                Ok(null_mut())
            }
            Op::TensorSize { t, dim } => {
                let mut info = TcfgTensorInfo::empty();
                let size = match capi::api() {
                    Some(api) if (api.tensor_info)(reg(regs, *t), &mut info) != 0 => info.size(*dim),
                    _ => None,
                };
                match size {
                    Some(size) => checked(ffi::PyLong_FromLongLong(size)),
                    None => {
                        set_error(
                            ffi::PyExc_SystemError,
                            "torch.vm: tensor_size ran without its guard".to_string(),
                        );
                        Err(())
                    }
                }
            }
            Op::TensorDim(t) => {
                let mut info = TcfgTensorInfo::empty();
                match capi::api() {
                    Some(api) if (api.tensor_info)(reg(regs, *t), &mut info) != 0 => {
                        checked(ffi::PyLong_FromLong(info.ndim as _))
                    }
                    _ => {
                        set_error(
                            ffi::PyExc_SystemError,
                            "torch.vm: tensor_dim ran without its guard".to_string(),
                        );
                        Err(())
                    }
                }
            }
            Op::Dispatch(d) => match capi::api() {
                Some(api) => self.exec_dispatch(regs, &d.slot, &d.operands, &d.fallback, api),
                None => self.exec_op(regs, &d.fallback),
            },
            Op::Guard(_) | Op::Graph(_) => Ok(null_mut()),
        }
    }

    // ------------------------------------------------------------------
    // Dispatch sites
    // ------------------------------------------------------------------

    /// The receiver of a call whose callee is `x.name` in the generic graph.
    fn method_receiver(&self, callee: V) -> Option<V> {
        match self.analysis.def_op(&self.func, callee) {
            Some(Op::LoadAttr(recv, _)) => Some(*recv),
            _ => None,
        }
    }

    /// The operands a site of the given kind takes for `op`, in the order the
    /// C++ side expects them.
    pub fn site_operands(&self, kind: i32, op: &Op) -> Option<Vec<V>> {
        match (kind, op) {
            (capi::TCFG_BINARY, Op::Binary(_, a, b)) | (capi::TCFG_BINARY, Op::Compare(_, a, b)) => {
                Some(vec![*a, *b])
            }
            (capi::TCFG_UNARY, Op::Unary(_, a))
            | (capi::TCFG_UNARY, Op::ToBool(a))
            | (capi::TCFG_UNARY, Op::LoadAttr(a, _)) => Some(vec![*a]),
            (capi::TCFG_CALL, Op::Call { callee, args, .. }) => {
                let mut v = vec![*callee];
                v.extend(args);
                Some(v)
            }
            (capi::TCFG_METHOD, Op::Call { callee, args, .. }) => {
                let mut v = vec![self.method_receiver(*callee)?];
                v.extend(args);
                Some(v)
            }
            (capi::TCFG_METHOD, Op::CallMethod { recv, args, .. }) => {
                let mut v = vec![*recv];
                v.extend(args);
                Some(v)
            }
            _ => None,
        }
    }

    /// The objects a site takes for the generic `op`. In generic code the
    /// receiver of a method call is read from the bound method, since its own
    /// register can be dead by the time of the call. `method` is the bound
    /// method of a CallMethod. Returns None if an operand is unbound.
    unsafe fn operand_ptrs(
        &self,
        regs: &mut [Obj],
        kind: i32,
        op: &Op,
        operands: &[V],
    ) -> Option<Vec<Obj>> {
        // The capture is matched against Python objects.
        let mut ptrs: Vec<Obj> = operands.iter().map(|v| self.obj(regs, *v)).collect();
        if let (capi::TCFG_METHOD, Op::Call { callee, .. }) = (kind, op) {
            let c = self.obj(regs, *callee);
            if c.is_null() || ffi::PyCFunction_Check(c) == 0 {
                return None;
            }
            ptrs[0] = ffi::PyCFunction_GetSelf(c);
        }
        ptrs.iter().all(|p| !p.is_null()).then_some(ptrs)
    }

    /// Decides whether `op`, about to run on the current registers, could be
    /// a single tensor operation, and of which kind.
    unsafe fn site_kind(&self, regs: &[Obj], op: &Op, api: &TcfgApi) -> Option<i32> {
        let is_tensor = |v: V| -> bool {
            let o = reg(regs, v);
            let mut info = TcfgTensorInfo::empty();
            !o.is_null() && (api.tensor_info)(o, &mut info) != 0
        };
        match op {
            Op::Binary(_, a, b) | Op::Compare(_, a, b) => {
                (is_tensor(*a) || is_tensor(*b)).then_some(capi::TCFG_BINARY)
            }
            Op::Unary(UnaryKind::Neg | UnaryKind::Pos | UnaryKind::Invert, a)
            | Op::ToBool(a)
            | Op::LoadAttr(a, _) => is_tensor(*a).then_some(capi::TCFG_UNARY),
            Op::Call { callee, .. } => {
                let c = reg(regs, *callee);
                if c.is_null() || capi::is_unboxed(c) || ffi::PyCFunction_Check(c) == 0 {
                    return None;
                }
                // x.f(...) on a tensor: the callee is a builtin bound to the
                // receiver. The receiver's register may already be released
                // (the bound method keeps the object alive), so ask the
                // method.
                let this = ffi::PyCFunction_GetSelf(c);
                let mut info = TcfgTensorInfo::empty();
                match self.method_receiver(*callee) {
                    Some(recv)
                        if !this.is_null()
                            && (api.tensor_info)(this, &mut info) != 0
                            && (reg(regs, recv).is_null() || reg(regs, recv) == this) =>
                    {
                        Some(capi::TCFG_METHOD)
                    }
                    _ => Some(capi::TCFG_CALL),
                }
            }
            Op::CallMethod { recv, .. } => is_tensor(*recv).then_some(capi::TCFG_METHOD),
            _ => None,
        }
    }

    /// Runs the generic `op` under a dispatch capture and tries to turn what
    /// was recorded into a call site. Returns the result of the operation and
    /// the site (or the reason there is none).
    unsafe fn capture_exec(
        &self,
        regs: &mut [Obj],
        op: &Op,
        kind: i32,
        api: &TcfgApi,
    ) -> (OpResult, *mut capi::TcfgSite, Option<String>, Kernels, Vec<String>) {
        let Some(operands) = self.site_operands(kind, op) else {
            return (self.exec_op(regs, op), null_mut(), None, Kernels::Unseen, Vec::new());
        };
        // Operands are borrowed from the registers, which the operation does
        // not touch; read them first so a call sees its callee.
        let ptrs_before = self.operand_ptrs(regs, kind, op, &operands);
        // What the operation calls is the interpreter's while it is looked
        // at: a Python function under it (`Tensor.__rsub__`) that is one
        // kernel call makes the operation that call, and compiled it would
        // be a function with a call site of its own.
        let nobody = Hooked::nobody();
        (api.capture_begin)();
        // A method call needs the bound method it looked up for the analysis.
        let mut method: Obj = null_mut();
        let result = match op {
            Op::CallMethod { recv, name, args, kwnames } => {
                method = ffi::PyObject_GetAttr(
                    self.obj(regs, *recv),
                    self.func.consts[*name as usize].as_ptr(),
                );
                if method.is_null() {
                    Err(())
                } else {
                    checked(self.vectorcall(regs, method, args, *kwnames))
                }
            }
            _ => self.exec_op(regs, op),
        };
        let trace = (api.capture_end)();
        drop(nobody);
        let mut site = null_mut();
        let mut reason = None;
        let mut operators = Vec::new();
        let kernels = Self::counted(trace, api);
        if !trace.is_null() {
            if let Ok(obj) = result {
                if let (false, Some(ptrs)) = (obj.is_null(), ptrs_before) {
                    let consts: Vec<u8> = operands
                        .iter()
                        .map(|v| matches!(self.analysis.def_op(&self.func, *v), Some(Op::Const(_))) as u8)
                        .collect();
                    let (callee, kwnames) = match op {
                        Op::Call { callee, kwnames, .. } => (self.obj(regs, *callee), *kwnames),
                        Op::CallMethod { kwnames, .. } => (method, *kwnames),
                        _ => (null_mut(), NONE),
                    };
                    let pyop = TcfgPyOp {
                        kind,
                        n_operands: ptrs.len() as i32,
                        operands: ptrs.as_ptr(),
                        is_const: consts.as_ptr(),
                        kwnames: if kwnames == NONE {
                            null_mut()
                        } else {
                            self.func.consts[kwnames as usize].as_ptr()
                        },
                        callee,
                        result: obj,
                        flags: if self.options.skip_autograd {
                            capi::TCFG_ALLOW_SKIP_AUTOGRAD
                        } else {
                            0
                        },
                    };
                    let mut buf = [0 as c_char; 160];
                    site = (api.site_from_trace)(trace, &pyop, buf.as_mut_ptr(), buf.len());
                    if site.is_null() {
                        reason = Some(
                            std::ffi::CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned(),
                        );
                    }
                }
            }
            if site.is_null() {
                operators = Self::called(trace, api);
            }
            (api.trace_free)(trace);
        }
        xdecref(method);
        (result, site, reason, kernels, operators)
    }

    /// The operators a capture saw the instruction call itself: not what
    /// those called in turn.
    unsafe fn called(trace: *mut capi::TcfgTrace, api: &TcfgApi) -> Vec<String> {
        let records = (api.trace_records)(trace);
        if records.is_null() || ffi::PyList_Check(records) == 0 {
            ffi::PyErr_Clear();
            xdecref(records);
            return vec![spec::UNKNOWN_OPERATOR.to_string()];
        }
        let mut names = Vec::new();
        for i in 0..ffi::PyList_GET_SIZE(records) {
            let record = ffi::PyList_GET_ITEM(records, i);
            let depth = ffi::PyLong_AsLong(ffi::PyTuple_GET_ITEM(record, 2));
            let again = ffi::PyObject_IsTrue(ffi::PyTuple_GET_ITEM(record, 3)) != 0;
            if depth == 0 && !again {
                names.push(pyffi::str_to_string(ffi::PyTuple_GET_ITEM(record, 0)));
            }
        }
        ffi::Py_DECREF(records);
        names
    }

    /// How many dispatcher calls a capture saw.
    unsafe fn counted(trace: *mut capi::TcfgTrace, api: &TcfgApi) -> Kernels {
        if trace.is_null() {
            return Kernels::None;
        }
        match (api.trace_calls)(trace) {
            0 => Kernels::None,
            1 => Kernels::One,
            _ => Kernels::Many,
        }
    }

    /// Calls the kernel of the first site whose guards accept the operands.
    /// When none does, the generic operation runs, under a capture so the
    /// slot can learn the new case.
    #[inline(always)]
    unsafe fn exec_dispatch(
        &self,
        regs: &mut [Obj],
        slot: &SiteSlot,
        operands: &[V],
        fallback: &Op,
        api: &TcfgApi,
    ) -> OpResult {
        let mut small: [Obj; 12] = [null_mut(); 12];
        let big: Vec<Obj>;
        let ptr: *const Obj = if operands.len() <= small.len() {
            for (i, v) in operands.iter().enumerate() {
                small[i] = reg(regs, *v);
            }
            small.as_ptr()
        } else {
            big = operands.iter().map(|v| reg(regs, *v)).collect();
            big.as_ptr()
        };
        let flags = if self.options.unboxed { capi::TCFG_UNBOXED_RESULT } else { 0 };
        self.exec_slot(regs, slot, ptr, flags, fallback, api)
    }

    /// `ptr` points at the operands, which may be unboxed tensors. With
    /// TCFG_UNBOXED_RESULT in `flags` the result may be one as well.
    unsafe fn exec_slot(
        &self,
        regs: &mut [Obj],
        slot: &SiteSlot,
        ptr: *const Obj,
        flags: u32,
        fallback: &Op,
        api: &TcfgApi,
    ) -> OpResult {
        let n = slot.len();
        for i in 0..n {
            let mut status = 0i32;
            let r = (api.site_call)(slot.get(i), ptr, flags, &mut status);
            if !r.is_null() {
                slot.hits.fetch_add(1, Ordering::Relaxed);
                if capi::is_unboxed(r) {
                    self.unboxed_results.fetch_add(1, Ordering::Relaxed);
                }
                return Ok(r);
            }
            if status != capi::TCFG_MISS {
                return Err(());
            }
        }
        let misses = slot.misses.fetch_add(1, Ordering::Relaxed);
        // The generic operation can be a call of anything.
        if self.unbound.any.get() {
            self.bind();
        }
        if n < self.options.max_site_variants && misses < MAX_CAPTURE_ATTEMPTS as u64 * 4 {
            let (result, site, _, _, _) = self.capture_exec(regs, fallback, slot.kind, api);
            if !site.is_null() {
                slot.push(site);
            }
            return result;
        }
        self.exec_op(regs, fallback)
    }

    /// One instruction in the profiling tier: it runs as the generic one
    /// does, and what it did is taken down. Whether it is one dispatcher
    /// call that can become a call site, and for every instruction what it
    /// was seen to do (`InstFacts`).
    unsafe fn exec_profiled(
        &self,
        b: usize,
        index: usize,
        inst: &Inst,
        regs: &mut [Obj],
        api: &TcfgApi,
    ) -> OpResult {
        let (slot, attempts, runs) = {
            let st = self.state.borrow();
            let bs = &st.blocks[b];
            (bs.slots[index].clone(), bs.capture_attempts[index], bs.facts[index].runs)
        };
        let outward = inst.op.calls_out();
        let probe = if outward { Some(Probe::start(&self.callees, regs, &inst.op, api)) } else { None };
        // A call of anything but a C function is not put under a capture:
        // what it runs has no end that is known here.
        let call = matches!(inst.op, Op::Call { .. } | Op::CallMethod { .. } | Op::CallEx { .. });
        let count = outward && !call && runs < MAX_COUNTED_RUNS;
        let mut kernels = if outward { Kernels::Unseen } else { Kernels::None };
        let mut operators = Vec::new();
        let result = if let Some(slot) = slot {
            let ptrs = self
                .site_operands(slot.kind, &inst.op)
                .and_then(|operands| self.operand_ptrs(regs, slot.kind, &inst.op, &operands));
            kernels = Kernels::One;
            match ptrs {
                Some(ptrs) => self.exec_slot(regs, &slot, ptrs.as_ptr(), 0, &inst.op, api),
                None => self.exec_op(regs, &inst.op),
            }
        } else {
            let kind = if attempts >= MAX_CAPTURE_ATTEMPTS { None } else { self.site_kind(regs, &inst.op, api) };
            match kind {
                None if count => {
                    (api.capture_begin)();
                    let result = self.exec_op(regs, &inst.op);
                    let trace = (api.capture_end)();
                    kernels = Self::counted(trace, api);
                    if !trace.is_null() {
                        operators = Self::called(trace, api);
                        (api.trace_free)(trace);
                    }
                    result
                }
                None => self.exec_op(regs, &inst.op),
                Some(kind) => {
                    let (result, site, reason, calls, names) = self.capture_exec(regs, &inst.op, kind, api);
                    kernels = calls;
                    operators = names;
                    let mut st = self.state.borrow_mut();
                    let bs = &mut st.blocks[b];
                    bs.capture_attempts[index] += 1;
                    if !site.is_null() {
                        let slot = SiteSlot::new(kind);
                        slot.push(site);
                        bs.slots[index] = Some(Arc::new(slot));
                        bs.reasons[index] = None;
                    } else if reason.is_some() {
                        bs.reasons[index] = reason;
                    }
                    result
                }
            }
        };
        let mut st = self.state.borrow_mut();
        let facts = &mut st.blocks[b].facts[index];
        facts.runs += 1;
        facts.saw_kernels(kernels);
        facts.saw_operators(operators);
        facts.raised |= result.is_err();
        if let Some(probe) = probe {
            probe.finish(&self.callees, facts, api);
        }
        if let Ok(obj) = result {
            if inst.dst != NONE {
                st.obs[inst.dst as usize].observe(obj);
            }
        }
        result
    }

    // ------------------------------------------------------------------
    // Guards, versions, specialization
    // ------------------------------------------------------------------

    #[inline(always)]
    unsafe fn guard_holds(cond: &GuardCond, regs: &[Obj], api: Option<&TcfgApi>) -> bool {
        let Some(api) = api else {
            return false;
        };
        let obj = reg(regs, cond.value());
        let mut info = TcfgTensorInfo::empty();
        if obj.is_null() || (api.tensor_info)(obj, &mut info) == 0 {
            return false;
        }
        match cond {
            GuardCond::TensorRank { min_rank, .. } => info.ndim as u32 >= *min_rank,
            GuardCond::DimRange { dim, lo, hi, .. } => {
                matches!(info.size(*dim), Some(size) if size >= *lo && size <= *hi)
            }
        }
    }

    #[inline(always)]
    unsafe fn select(&self, b: usize, regs: &[Obj], api: Option<&TcfgApi>) -> (*const BlockCode, Mode) {
        let slot = self.settled.blocks.get_unchecked(b);
        let settled = slot.get();
        if !settled.code.is_null() {
            slot.set(Settled { entries: settled.entries + 1, ..settled });
            return (settled.code, if settled.specialized { Mode::Spec } else { Mode::Generic });
        }
        self.select_slow(b, regs, api)
    }

    unsafe fn select_slow(&self, b: usize, regs: &[Obj], api: Option<&TcfgApi>) -> (*const BlockCode, Mode) {
        let generic = Arc::as_ptr(self.generic.get_unchecked(b));
        let settle = |code: *const BlockCode, specialized: bool| {
            self.settled.blocks.get_unchecked(b).set(Settled { code, specialized, entries: 0 });
        };
        if !self.options.specialize || api.is_none() {
            settle(generic, false);
            return (generic, Mode::Generic);
        }
        let mut st = self.state.borrow_mut();
        let bs = &mut st.blocks[b];
        bs.execs += 1;
        // The copy of a version that enters its graphs runs in its place
        // while one of them has something to do. `enter_graph` has the
        // block looked at again when one has not.
        let entered = |version: &Arc<BlockCode>| {
            let copy = version.entering.as_ref().filter(|copy| copy.graphs.iter().any(|site| site.runs()));
            copy.map_or(Arc::as_ptr(version), Arc::as_ptr)
        };
        // One version that is entered unconditionally, or none and no more
        // attempts at one: from now on there is nothing to decide.
        if bs.versions.len() == 1 && bs.versions[0].entry_guards.is_empty() {
            settle(entered(&bs.versions[0]), true);
        } else if bs.versions.is_empty() && bs.respec_left == 0 {
            settle(generic, false);
        }
        for version in &bs.versions {
            if version.entry_guards.iter().all(|g| Self::guard_holds(g, regs, api)) {
                bs.spec_execs += 1;
                return (entered(version), Mode::Spec);
            }
        }
        if !bs.versions.is_empty() {
            bs.entry_misses += 1;
        }
        let mode = if bs.respec_left > 0 { Mode::Profile } else { Mode::Generic };
        if mode == Mode::Profile {
            // What the block is entered with is part of its profile.
            let live: Vec<V> = self.analysis.live_in[b].iter().collect();
            for v in live {
                st.obs[v as usize].observe(reg(regs, v));
            }
        }
        (generic, mode)
    }

    /// How often block `b` was entered, and how often in a specialized
    /// version, counting the entries that did not go through the state.
    pub fn block_entries(&self, b: usize, bs: &BlockState) -> (u64, u64) {
        let settled = self.settled.blocks[b].get();
        let specialized = if settled.specialized { settled.entries } else { 0 };
        (bs.execs + settled.entries, bs.spec_execs + specialized)
    }

    pub fn calls(&self) -> u64 {
        self.settled.calls.get()
    }

    /// Called when a block finished a run in the profiling tier: runs the
    /// pass pipeline over a fresh copy of the block and installs the result
    /// as a new version.
    unsafe fn specialize(&self, b: usize) {
        // The passes are not the function's code.
        let _nobody = Hooked::nobody();
        let (slots, obs, facts, reasons, speculate, id) = {
            let mut st = self.state.borrow_mut();
            let obs = st.obs.clone();
            let bs = &mut st.blocks[b];
            if bs.respec_left == 0 {
                return;
            }
            bs.respec_left -= 1;
            bs.profile_runs += 1;
            bs.raised = 0;
            let id = bs.next_version_id;
            bs.next_version_id += 1;
            // The last attempt must hold for every input.
            (bs.slots.clone(), obs, bs.facts.clone(), bs.reasons.clone(), bs.respec_left > 0, id)
        };
        let mut unit = Unit::from_generic(&self.func, b, speculate);
        let cx = PassCx { cf: self, slots: &slots, obs: &obs, facts: &facts, reasons: &reasons };
        let outcome = match self.pipeline.run(&mut unit, &cx) {
            Ok(true) => spec::finalize(unit, self, id).map(Some),
            Ok(false) => Ok(None),
            Err(err) => {
                // A failing pass must not take the program down with it: the
                // block simply stays generic.
                let msg = pyo3::Python::attach(|py| err.value(py).to_string());
                Err(format!("pass failed: {}", msg))
            }
        };
        let mut st = self.state.borrow_mut();
        let unsettled = Settled { code: std::ptr::null(), ..self.settled.blocks[b].get() };
        self.settled.blocks[b].set(unsettled);
        match outcome {
            Ok(Some(code)) => {
                let guarded = !code.entry_guards.is_empty()
                    || code.insts.iter().any(|i| matches!(i.op, Op::Guard(_)));
                let bs = &mut st.blocks[b];
                bs.versions.insert(0, Arc::new(code));
                // Nothing can send an unguarded version back to profiling.
                let keep = if guarded { self.options.max_versions } else { 1 };
                if bs.versions.len() > keep {
                    let replaced = bs.versions.split_off(keep);
                    bs.retired.extend(replaced);
                }
            }
            Ok(None) => st.blocks[b].respec_left = 0,
            Err(msg) => {
                st.blocks[b].respec_left = 0;
                st.blocks[b].rejected = Some(msg.clone());
                st.log.push(format!("b{}: {}", b, msg));
            }
        }
    }

    /// Called when an instruction raised in a profiling run of a block. A
    /// block that raises every time it runs (a lookup that fails, and is
    /// caught) would profile forever, and so would hold back all that waits
    /// for the profiling above it to be over: the second such run in a row
    /// is specialized from as one that got to the end is.
    unsafe fn raised_profiling(&self, b: usize) {
        let enough = {
            let mut st = self.state.borrow_mut();
            let bs = &mut st.blocks[b];
            bs.raised += 1;
            bs.raised >= PROFILE_RAISES
        };
        if enough {
            // The passes run with no error set.
            let raised = ffi::PyErr_GetRaisedException();
            self.specialize(b);
            ffi::PyErr_SetRaisedException(raised);
        }
    }

    // ------------------------------------------------------------------
    // Graphs
    // ------------------------------------------------------------------

    /// Runs `Op::Graph`, which stands in front of the first operation of a
    /// graph, at instruction `pc - 1` of `code`. If one of the paths that
    /// were taken down from here is the one this call takes, and one of the
    /// forms that were made of it is for the values of the call, its
    /// callable runs in place of the nodes and the function goes on where
    /// the path ends. If none is and the graph may get another, the
    /// instructions run here and are taken down as a path. Otherwise
    /// nothing happens. `serial` tells the calls of the function apart.
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn enter_graph(
        &self,
        site: &Arc<GraphSite>,
        regs: &mut [Obj],
        frames: &mut Frames,
        api: &TcfgApi,
        block: usize,
        code: *const BlockCode,
        pc: usize,
        serial: u64,
    ) -> PathEnd {
        if !site.runs() {
            // The code that enters it is not needed for it anymore.
            let unsettled = Settled { code: std::ptr::null(), ..self.settled.blocks[block].get() };
            self.settled.blocks[block].set(unsettled);
            return PathEnd::Miss;
        }
        if site.missed.load(Ordering::Relaxed) == serial {
            return PathEnd::Miss;
        }
        let modes = (api.call_modes)();
        let (mut included, mut excluded) = (0u64, 0u64);
        (api.dispatch_keys)(&mut included, &mut excluded);
        let world = (modes, included, excluded);
        // What else a compiled graph depends on is asked of Python: only
        // for a call that is like one a form was made for in all the rest.
        let mut state: Option<u64> = None;
        for (at, path) in site.paths().into_iter().enumerate() {
            match self.run_path(site, &path, regs, frames, api, world, &mut state) {
                Ran::Landed(landing) => {
                    site.hits.fetch_add(1, Ordering::Relaxed);
                    if at > 0 {
                        let mut paths = site.variants.borrow_mut();
                        if paths.get(at).is_some_and(|other| Arc::ptr_eq(other, &path)) {
                            let ran = paths.remove(at);
                            paths.insert(0, ran);
                        }
                    }
                    return PathEnd::Landed(landing);
                }
                Ran::Miss => {
                    site.misses.fetch_add(1, Ordering::Relaxed);
                    return PathEnd::Miss;
                }
                Ran::Other => {}
            }
        }
        // A function further out that is still being profiled may get a
        // graph that holds these operations, and so may a graph that starts
        // earlier and is not done being taken down: that one goes first.
        if WARMING.with(|w| w.get()) > self.settled.warming.get() || profiling_depth() > 0 {
            // A call in which a graph waits says nothing about how fast the
            // function is going to be: it is no sample of a measurement.
            unsettled();
            return PathEnd::Miss;
        }
        let waits = {
            let st = self.state.borrow();
            let mut others = st.blocks.iter().flat_map(|bs| bs.graphs.iter().map(|(_, other)| other));
            others.any(|other| !Arc::ptr_eq(other, site) && other.pending())
        };
        if waits {
            unsettled();
            return PathEnd::Miss;
        }
        if site.exporter.is_none() && site.variants.borrow().is_empty() && self.asks(site, block) {
            // A graph that starts earlier in the function is taken down
            // again: this call goes a way its path does not, and may come
            // through here.
            return PathEnd::Miss;
        }
        site.misses.fetch_add(1, Ordering::Relaxed);
        if site.records() {
            let state = state.unwrap_or_else(|| graph::thread_state());
            return self.record(site, regs, frames, api, block, code, pc, (modes, included, excluded, state));
        }
        // The loop that comes here again in this call finds the same.
        site.missed.store(serial, Ordering::Relaxed);
        PathEnd::Miss
    }

    /// For a graph that is not exported and has no path yet: whether a
    /// graph that starts earlier in the function and has its path is asked
    /// to take another one down, which a graph does twice at most. Where
    /// graphs are exported, the earlier one finds out itself that a call is
    /// not on its path.
    unsafe fn asks(&self, site: &Arc<GraphSite>, block: usize) -> bool {
        let earlier: Option<(usize, Arc<GraphSite>)> = {
            let st = self.state.borrow();
            let before = st.blocks[..=block].iter().enumerate();
            before
                .flat_map(|(b, bs)| bs.graphs.iter().map(move |(_, other)| (b, other.clone())))
                .find(|(_, other)| !Arc::ptr_eq(other, site) && other.asked())
        };
        let Some((at, _)) = earlier else {
            return false;
        };
        let unsettled = Settled { code: std::ptr::null(), ..self.settled.blocks[at].get() };
        self.settled.blocks[at].set(unsettled);
        true
    }

    /// Whether the values of a call are what an exported form was made for.
    /// The tensors it looks at have their Python objects afterwards.
    unsafe fn accepts(&self, exported: &Exported, regs: &mut [Obj], api: &TcfgApi) -> bool {
        let mut numbers: Vec<i64> = Vec::with_capacity(32);
        for check in &exported.checks {
            let o = check.target.at(regs);
            let holds = !o.is_null()
                && match &check.cond {
                    Cond::Is(expected) => o == expected.as_ptr(),
                    Cond::ValueEq(expected) => !capi::is_unboxed(o) && graph::same_value(o, expected.as_ptr()),
                    Cond::Tensor(row) => row.fits(o, &mut numbers, api),
                    Cond::Type(ty) => !capi::is_unboxed(o) && ffi::Py_TYPE(o) as Obj == ty.as_ptr(),
                    Cond::Sequence(ty, count) => {
                        !capi::is_unboxed(o)
                            && ffi::Py_TYPE(o) as usize == *ty
                            && ffi::PyObject_Size(o) as usize == *count
                    }
                    Cond::Has(name, expected) => {
                        let found = if graph::quiet_attr(o, name.as_ptr(), api) {
                            ffi::PyObject_GetAttr(o, name.as_ptr())
                        } else {
                            null_mut()
                        };
                        ffi::PyErr_Clear();
                        xdecref(found);
                        found == expected.as_ptr()
                    }
                };
            if !holds {
                return false;
            }
        }
        if !exported.sizes.holds_numbers(&numbers) {
            if graph::debug() {
                eprintln!("torch.vm graph: the sizes of a form do not hold");
            }
            return false;
        }
        // With their Python objects: the class of each tensor, and which of
        // them are one and the same.
        let mut seen: Vec<Obj> = Vec::new();
        for check in &exported.checks {
            let Cond::Tensor(row) = &check.cond else {
                continue;
            };
            // What a list holds has its Python object.
            let o = match check.target.item {
                NONE => self.obj(regs, check.target.value),
                _ => check.target.at(regs),
            };
            let ty = ffi::Py_TYPE(o);
            let same = seen.iter().position(|other| *other == o).unwrap_or(seen.len());
            if ty as usize != row.ty || (*ty).tp_version_tag != row.type_version || same != row.same {
                if graph::debug() {
                    let (class, version) = (ty as usize != row.ty, (*ty).tp_version_tag != row.type_version);
                    eprintln!("torch.vm graph: a tensor of a form is of another class ({}), version of its class ({}), or the same as another ({} for {})", class, version, same, row.same);
                }
                return false;
            }
            seen.push(o);
        }
        true
    }

    /// Why a recorded run with writes in place is not exported, if what
    /// autograd saved of its operations is not what it was, or cannot be
    /// shown to be (asked of `torch.vm._export.saved_changed`): a tensor
    /// that an operation saved for its derivative and that a later one
    /// wrote to in place makes eager raise when it goes backward.
    unsafe fn saved_changed(&self, path: &Path, rec: &graph::Recording) -> Option<String> {
        let results = ffi::PyList_New(0);
        for inst in path.body.iter().filter(|inst| inst.kind == Kind::Node && inst.dst != NONE) {
            let held = rec.values[inst.dst as usize];
            if !held.is_null() {
                ffi::PyList_Append(results, held);
            }
        }
        let _nobody = Hooked::nobody();
        let module = ffi::PyImport_ImportModule(c"torch.vm._export".as_ptr());
        let said = if module.is_null() { null_mut() } else { ffi::PyObject_CallMethod(module, c"saved_changed".as_ptr(), c"O".as_ptr(), results) };
        let why = if said.is_null() || ffi::PyUnicode_Check(said) == 0 {
            Some("what autograd saved could not be asked for".to_string())
        } else {
            Some(pyffi::str_to_string(said)).filter(|why| !why.is_empty())
        };
        ffi::PyErr_Clear();
        for held in [results, module, said] {
            xdecref(held);
        }
        why
    }

    /// Whether an instruction of a path runs no Python code and writes to
    /// nothing but what the path made, on the values it has now: what an
    /// exported form has run in front of its callable or behind it, and so
    /// not where the function has it, and again when the call turns out to
    /// be one the path is not for. A read of a variable; arithmetic on
    /// numbers; an attribute that is found without a property or a
    /// `__getattr__` of Python; an item of a list, a tuple or a dict; a list
    /// or a dict that is made; a loop over a list, a tuple or a range.
    unsafe fn quiet(&self, regs: &[Obj], op: &Op, api: &TcfgApi) -> bool {
        self.quiet_of(regs, op, &|_| true, api)
    }

    /// `quiet` where a path is run: `own` says of a register whether the
    /// path made what it holds. What goes over an iterator uses it up,
    /// and may do that to one the path made at this call and to no other:
    /// a value that was a list when the path was taken down may be an
    /// iterator of the caller's now. (While a path is taken down, `record`
    /// asks that of the trace itself.)
    unsafe fn quiet_of(&self, regs: &[Obj], op: &Op, own: &dyn Fn(V) -> bool, api: &TcfgApi) -> bool {
        let at = |v: V| reg(regs, v);
        match op {
            Op::Const(_)
            | Op::LoadGlobal(_)
            | Op::LoadDeref(_)
            | Op::CellGet(..)
            | Op::TensorSize { .. }
            | Op::TensorDim(_)
            | Op::TupleGet(..)
            | Op::IsNone(..)
            | Op::Is(..)
            | Op::IsFunction { .. }
            | Op::Part(..)
            | Op::Object(_)
            | Op::CheckBound(..) => true,
            // Of what is there: a register that is empty is nothing to
            // build of.
            Op::BuildTuple(vs) | Op::BuildList(vs) => vs.iter().all(|v| !at(*v).is_null()),
            Op::BuildSlice(a, b, c) => [a, b, c].into_iter().all(|v| *v == NONE || !at(*v).is_null()),
            // What changes a list, or draws on an iterator, does so to one
            // the path made: `record` takes down no other.
            Op::ListAppend(..) | Op::IterNext(_) => true,
            Op::ListToTuple(l) => !at(*l).is_null() && ffi::PyList_CheckExact(at(*l)) != 0,
            // Onto a list, or into a dict, that the path made.
            Op::ListExtend(l, more) => {
                let plain = |o: Obj| !o.is_null() && !capi::is_unboxed(o);
                plain(at(*l))
                    && plain(at(*more))
                    && ffi::PyList_CheckExact(at(*l)) != 0
                    && (ffi::PyList_CheckExact(at(*more)) != 0 || ffi::PyTuple_CheckExact(at(*more)) != 0)
            }
            Op::DictMerge(d, more, _) => graph::quiet_merge(at(*d), at(*more)),
            Op::ImportName { name, fromlist, level } => {
                let f = &self.func;
                graph::quiet_import(f.builtins.as_ptr(), f.consts[*name as usize].as_ptr(), at(*fromlist), at(*level))
            }
            Op::BuildMap(vs) => vs.iter().step_by(2).all(|k| !at(*k).is_null() && graph::is_plain_key(at(*k))),
            Op::MapAdd(_, k, _) => !at(*k).is_null() && graph::is_plain_key(at(*k)),
            Op::GetIter(v) => !at(*v).is_null() && graph::quiet_iter(at(*v)).is_some_and(|used_up| !used_up || own(*v)),
            Op::Binary(_, a, b) => [a, b].iter().all(|v| !at(**v).is_null() && graph::is_number(at(**v))),
            Op::Unary(UnaryKind::Not, a) | Op::ToBool(a) => !at(*a).is_null() && graph::quiet_truth(at(*a)),
            Op::Unary(_, a) => !at(*a).is_null() && graph::is_number(at(*a)),
            Op::Compare(_, a, b) => !at(*a).is_null() && !at(*b).is_null() && graph::quiet_compare(at(*a), at(*b)),
            Op::LoadAttr(a, name) => {
                !at(*a).is_null() && graph::quiet_attr(at(*a), self.func.consts[*name as usize].as_ptr(), api)
            }
            Op::GetItem(a, k) => !at(*a).is_null() && !at(*k).is_null() && graph::quiet_item(at(*a), at(*k)),
            Op::Contains(item, container, _) => {
                !at(*item).is_null() && !at(*container).is_null() && graph::quiet_contains(at(*item), at(*container))
            }
            Op::GetLen(v) | Op::Unpack(v, _) => !at(*v).is_null() && graph::quiet_sized(at(*v)),
            // An iterator among the arguments is gone over only if the
            // path made it.
            Op::Call { callee, args, kwnames } => {
                let given: Vec<Obj> = args.iter().map(|v| at(*v)).collect();
                *kwnames == NONE && graph::pure_call(at(*callee), &given, |i| own(args[i]), api).is_some()
            }
            Op::CallMethod { recv, name, args, kwnames } => {
                let given: Vec<Obj> = args.iter().map(|v| at(*v)).collect();
                *kwnames == NONE && graph::pure_method(at(*recv), self.func.consts[*name as usize].as_ptr(), &given, api)
            }
            _ => false,
        }
    }

    /// Runs the instructions of one stage of a path, and its checks, on the
    /// registers of the path. `function` is the function that is running,
    /// for its closure.
    unsafe fn run_stage(
        &self,
        site: &GraphSite,
        path: &Path,
        stage: Stage,
        own: &mut [Obj],
        function: Obj,
        api: &TcfgApi,
    ) -> Staged {
        for inst in path.staged(stage) {
            let owner = &*path.owners[inst.owner as usize];
            if let Some(assert) = &inst.assert {
                let holds = match assert {
                    Assert::Truth(v, way) => reg(own, *v) == if *way { ffi::Py_True() } else { ffi::Py_False() },
                    Assert::Null(v, exhausted) => reg(own, *v).is_null() == *exhausted,
                    Assert::Guard(cond) => Self::guard_holds(cond, own, Some(api)),
                    Assert::Enters(v, compiled, defaults) => {
                        let function = reg(own, *v);
                        !function.is_null() && self.follows(function) == Some(*compiled) && Self::defaults_of(function) == *defaults
                    }
                    Assert::Iterator(v) => !reg(own, *v).is_null() && ffi::PyIter_Check(reg(own, *v)) != 0,
                    Assert::Count(v) => !reg(own, *v).is_null() && graph::is_count(reg(own, *v)),
                    Assert::Absent(v, name) => graph::absent(reg(own, *v), name.as_ptr()),
                    Assert::Lacks(v, name) => {
                        let held = reg(own, *v);
                        !held.is_null() && !capi::is_unboxed(held) && pyffi::_PyType_Lookup(ffi::Py_TYPE(held), name.as_ptr()).is_null()
                    }
                    Assert::Sequence(v, class, count) => {
                        let held = reg(own, *v);
                        !held.is_null()
                            && !capi::is_unboxed(held)
                            && ffi::Py_TYPE(held) as usize == *class
                            && ffi::PyObject_Size(held) as usize == *count
                    }
                    Assert::Equals(v, number) => !reg(own, *v).is_null() && graph::same_value(reg(own, *v), number.as_ptr()),
                    Assert::Keys(v, names) => {
                        let held = reg(own, *v);
                        let dict = !held.is_null() && !capi::is_unboxed(held) && ffi::PyDict_CheckExact(held) != 0;
                        dict && ffi::PyDict_Size(held) as usize == names.len() && {
                            let (mut at, mut key, mut value) = (0, null_mut(), null_mut());
                            names.iter().all(|name| {
                                ffi::PyDict_Next(held, &mut at, &mut key, &mut value) != 0 && key == name.as_ptr()
                            })
                        }
                    }
                };
                if !holds {
                    if graph::debug() {
                        let at = path.body.iter().position(|other| std::ptr::eq(other, inst)).unwrap_or(0);
                        let what = match assert {
                            Assert::Truth(..) => "the way of a branch",
                            Assert::Null(..) => "the end of a loop",
                            Assert::Guard(_) => "a guard of the code",
                            Assert::Enters(..) => "the function a call enters",
                            Assert::Iterator(_) | Assert::Count(_) => "what a function of a class returned",
                            Assert::Absent(..) => "an attribute left to __getattr__",
                            Assert::Lacks(..) => "a class without a __len__",
                            Assert::Sequence(..) | Assert::Keys(..) => "what a call spreads",
                            Assert::Equals(..) => "the place of an item",
                        };
                        let made = path.body.iter().find(|other| other.dst == assert.value() && other.dst != NONE);
                        let from = made.map_or(String::new(), |made| (&*path.owners[made.owner as usize]).func.op_repr(&made.op));
                        eprintln!(
                            "torch.vm graph {}: a check of the path does not hold: {} (instruction {} of {}, in {}; of: {})",
                            site.name, what, at, path.body.len(), owner.func.qualname, from
                        );
                    }
                    return Staged::Failed;
                }
                continue;
            }
            let result = match &inst.op {
                // The registers of a path do not have the function.
                Op::LoadDeref(i) => {
                    let closure = ffi::PyFunction_GetClosure(function);
                    let there = !closure.is_null() && ffi::PyTuple_GET_SIZE(closure) > *i as isize;
                    let held = if there { pyffi::PyCell_Get(ffi::PyTuple_GET_ITEM(closure, *i as isize)) } else { null_mut() };
                    checked(held)
                }
                op if owner.quiet_of(own, op, &|v| path.made.get(v as usize).is_some_and(|made| *made), api) => owner.exec_op(own, op),
                // An attribute that the path read is not there at this
                // call, and `object` says so itself: the read raises and
                // does nothing else, where the instructions get to it.
                Op::LoadAttr(of, name) if graph::plainly_absent(reg(own, *of), owner.func.consts[*name as usize].as_ptr()) => {
                    return Staged::Failed;
                }
                op => {
                    let text = owner.func.op_repr(op);
                    site.unsure(text.split_whitespace().next().unwrap_or("an instruction"));
                    return Staged::Unsure;
                }
            };
            match result {
                Ok(obj) if inst.dst != NONE => capi::release(std::mem::replace(&mut own[inst.dst as usize], obj)),
                Ok(obj) => capi::release(obj),
                // The instructions raise it where they get to it.
                Err(()) => {
                    if graph::debug() {
                        eprintln!("torch.vm graph {}: {} of the path raises: {}", site.name, owner.func.op_repr(&inst.op), graph::raised_text());
                    }
                    ffi::PyErr_Clear();
                    return Staged::Failed;
                }
            }
        }
        Staged::Done
    }

    /// Whether an object stays alive where a path lands, whatever the path
    /// lets go of: a parameter of the frame that the path does not bind
    /// anew holds it, or a register of the function that the path neither
    /// writes nor clears. `frame` is the frame on top and whether it is
    /// the function's own (inside a call that is inlined it is that
    /// call's, and its variables are not the function's parameters).
    unsafe fn kept(&self, path: &Path, o: Obj, regs: &[Obj], frame: (Frame, bool)) -> bool {
        let (frame, own) = frame;
        if own && !frame.locals.is_null() {
            for slot in 0..self.func.params.len() {
                let bound = path.bound.iter().any(|(other, _)| *other as usize == slot);
                if !bound && frame_owned(*frame.locals.add(slot)) == o {
                    return true;
                }
            }
        }
        (0..self.func.nvalues as usize).any(|v| {
            regs[v] == o && !path.clears.contains(&(v as V)) && !path.writes.iter().any(|(_, to)| *to as usize == v)
        })
    }

    /// What an instruction lets go of when it runs while a path is taken
    /// down, added to `suspects`: what the registers it clears hold, what
    /// the register it writes holds, and what a variable it binds holds.
    /// True where one of them is held by
    /// nothing but what is let go of here and is not kept by the trace:
    /// it ends for good, and may take the last reference of the function
    /// to something the trace keeps with it.
    unsafe fn lets_go(&self, t: &Trace, inst: &Inst, regs: &[Obj], frame: Frame, suspects: &mut Vec<Obj>) -> bool {
        let from = suspects.len();
        let mut take = |held: Obj| {
            if !held.is_null() && !capi::is_unboxed(held) {
                suspects.push(held);
            }
        };
        for v in &inst.kills {
            take(reg(regs, *v));
        }
        if inst.dst != NONE {
            take(reg(regs, inst.dst));
        }
        if let (Op::StoreLocal(slot, _), false) = (&inst.op, frame.locals.is_null()) {
            take(frame_owned(*frame.locals.add(*slot as usize)));
        }
        Self::ends_for_good(t, &suspects[from..])
    }

    /// Whether one of the objects that are about to be let go of is kept
    /// by nothing else, not by the trace either, and refers to others.
    unsafe fn ends_for_good(t: &Trace, gone: &[Obj]) -> bool {
        gone.iter().any(|o| {
            let dropped = gone.iter().filter(|other| *other == o).count() as isize;
            !t.rec.kept.contains_key(&(*o as usize)) && ffi::Py_REFCNT(*o) <= dropped && (*ffi::Py_TYPE(*o)).tp_traverse.is_some()
        })
    }

    /// Asked where a function has let go of references while a path is
    /// taken down (`suspects`: the objects; `sweep`: one of them ended
    /// and may have taken further references with it). An object that
    /// the trace keeps, and that nothing else holds anymore, has ended
    /// for the function: it lives on only until the trace is let go of.
    /// Where the end of such an object may run Python code
    /// (`graph::end_quietly`, with the references of the trace counted
    /// out), that code has to run here, where the instructions end the
    /// object: the trace lets go of all it keeps, nothing is made of this
    /// run, and the path ends behind the instruction. True where that
    /// was done.
    unsafe fn overdue(&self, t: &mut Trace, suspects: &mut Vec<Obj>, sweep: bool) -> bool {
        if t.abandoned.is_some() {
            suspects.clear();
            return true;
        }
        let alone = |o: Obj, t: &Trace| t.rec.kept.get(&(o as usize)).is_some_and(|(count, _)| ffi::Py_REFCNT(o) <= *count);
        let mut ended: Vec<Obj> = if sweep {
            t.rec.kept.keys().map(|o| *o as Obj).filter(|o| alone(*o, t)).collect()
        } else {
            suspects.iter().copied().filter(|o| alone(*o, t)).collect()
        };
        suspects.clear();
        if ended.is_empty() {
            return false;
        }
        ended.sort_unstable();
        ended.dedup();
        // The function may be on its way out with an exception (a call
        // that raised has let go of what it held): the look asks things
        // of objects, and leaves what was raised as it is.
        let raised = ffi::PyErr_GetRaisedException();
        let found = {
            let mut kept = graph::Kept { kept: &t.rec.kept, defs: &t.defs };
            graph::end_quietly(&mut [], &[], &ended, Some(&mut kept))
        };
        if !raised.is_null() {
            ffi::PyErr_SetRaisedException(raised);
        }
        let Err(found) = found else {
            return false;
        };
        let class = if found.is_null() { "of which there are too many to look at".to_string() } else { format!("of the class {}", pyffi::type_name(found)) };
        let why = format!("the function lets go of an object {} whose end may run Python code", class);
        if graph::debug() {
            eprintln!("torch.vm graph: a path of {} is not taken down in this call: {}", self.func.qualname, why);
        }
        t.abandon(why);
        true
    }

    /// The first look at what a path lets go of where it lands, before it
    /// has taken anything of the call: what the registers of the function
    /// hold that it overwrites or clears, and what the variables of the
    /// frame hold that it binds anew. No reference to these objects is the
    /// path's yet, so the counts of references say exactly who else holds
    /// them: an object with a reference besides those the path lets go of
    /// stays alive, whatever it is and whatever it refers to; one the
    /// path hands on stays (`Path::handed`); the others end, and are
    /// looked at as they are at this call (`graph::end_quietly`). The
    /// instructions of the path and the callable let go of nothing that
    /// was there before them, so what is found here holds where the path
    /// lands.
    unsafe fn lets_go_early(&self, path: &Path, regs: &[Obj], frame: Frame) -> bool {
        let mut gone: Vec<Obj> = Vec::new();
        let mut take = |held: Obj| {
            if !held.is_null() && !capi::is_unboxed(held) {
                gone.push(held);
            }
        };
        for v in &path.clears {
            take(reg(regs, *v));
        }
        for (_, v) in &path.writes {
            take(reg(regs, *v));
        }
        if !frame.locals.is_null() {
            for (slot, _) in &path.bound {
                take(frame_owned(*frame.locals.add(*slot as usize)));
            }
        }
        if gone.is_empty() {
            return true;
        }
        let stay: Vec<Obj> = path.handed.iter().map(|v| reg(regs, *v)).collect();
        graph::end_quietly(&mut gone, &stay, &[], None).is_ok()
    }

    /// Whether everything a path lets go of where it lands can be let go
    /// of there. The instructions let go of a reference in their place; a
    /// path does where it lands, behind everything it read, so an object
    /// whose end runs Python code must not end by it. What a path lets go
    /// of that was there before it ran: what the registers of the
    /// function hold that it overwrites or clears, and what the variables
    /// of the frame hold that it binds anew. They were looked at before
    /// the path took anything (`early`: what `lets_go_early` said). Where
    /// that look said no (it takes a register that the path writes for
    /// one that is let go of, also where the path writes back what was
    /// there), everything is looked at once more, on the registers of the
    /// path: each object dies quietly, or stays alive because a parameter
    /// or an untouched register of the function holds it (`kept`) or the
    /// path hands it on (`Path::handed`); what is left is counted with
    /// the references the path itself holds (`graph::end_quietly`,
    /// `graph::Scan`).
    unsafe fn lets_all_go(&self, path: &Path, own: &[Obj], regs: &[Obj], frame: (Frame, bool), early: bool) -> bool {
        let mut gone: Vec<Obj> = Vec::new();
        let take = |gone: &mut Vec<Obj>, held: Obj| {
            if !held.is_null() && !capi::is_unboxed(held) {
                gone.push(held);
            }
        };
        if early {
            return true;
        }
        for v in &path.clears {
            take(&mut gone, reg(regs, *v));
        }
        for (made, v) in &path.writes {
            if reg(regs, *v) != reg(own, *made) {
                take(&mut gone, reg(regs, *v));
            }
        }
        if !frame.0.locals.is_null() {
            for (slot, _) in &path.bound {
                take(&mut gone, frame_owned(*frame.0.locals.add(*slot as usize)));
            }
        }
        // The objects that the cheap proofs say nothing of: not known to
        // end quietly with all they refer to, not kept by a parameter or
        // a register of the function, not handed on by the path.
        let mut open: Vec<Obj> = Vec::new();
        for (at, o) in gone.iter().enumerate() {
            let known = gone[..at].contains(o)
                || graph::dies_quietly(*o)
                || self.kept(path, *o, regs, frame)
                || path.handed.iter().any(|v| reg(regs, *v) == *o);
            if !known {
                open.push(*o);
            }
        }
        if open.is_empty() {
            return true;
        }
        // Those are looked at together, with the references the path
        // itself holds counted. Every reference that is let go of
        // counts; what the cheap proofs covered ends quietly with all it
        // refers to, or not at all, whatever else ends.
        let given: Vec<Obj> = path.inputs.iter().map(|(_, made)| reg(own, *made)).collect();
        let stay: Vec<Obj> = gone.iter().copied().filter(|o| !open.contains(o)).collect();
        let mut scan = graph::Scan::new(own, &given);
        let fine = graph::end_quietly(&mut gone, &stay, &[], Some(&mut scan));
        if let (Err(found), true) = (fine, graph::debug()) {
            eprintln!(
                "torch.vm graph: a path of {} is not for this call: where it lands it would let go of an object of the class {}, whose end may run Python code",
                self.func.qualname,
                pyffi::type_name(if found.is_null() { open[0] } else { found })
            );
        }
        fine.is_ok()
    }

    /// Runs a path in place of the instructions it was taken down from, if
    /// it is the one this call takes and one of its exported forms is for
    /// the values of the call. On registers of its own, which get the values
    /// it reads of the function's: the instructions that depend on no node,
    /// with the checks on the ways the branches went; then the checks of
    /// the form, the callable, and the instructions that work on what it
    /// returned. Only then the registers of the function are made what they
    /// are where the path ends: until there, a call the path is not for has
    /// changed nothing.
    unsafe fn run_path(
        &self,
        site: &GraphSite,
        path: &Path,
        regs: &mut [Obj],
        frames: &mut Frames,
        api: &TcfgApi,
        world: (u64, u64, u64),
        state: &mut Option<u64>,
    ) -> Ran {
        let of_world = |exported: &Exported| (exported.modes, exported.included, exported.excluded) == world;
        let count = path.exported.borrow().len();
        if !path.exported.borrow().iter().any(|exported| of_world(exported)) {
            return Ran::Other;
        }
        // What the path lets go of where it lands is looked at before it
        // holds anything itself.
        let early = self.lets_go_early(path, regs, frames.top());
        let mut own = path.registers.borrow_mut().pop().unwrap_or_else(|| vec![null_mut(); path.nregs]);
        for (v, made) in &path.inputs {
            let held = reg(regs, *v);
            capi::retain(held);
            own[*made as usize] = held;
        }
        let function = reg(regs, self.func.nvalues);
        // Lets go of what the registers of the path hold.
        let leave = |mut own: Vec<Obj>| {
            for held in own.iter_mut() {
                capi::release(std::mem::replace(held, null_mut()));
            }
            path.registers.borrow_mut().push(own);
        };
        let other = |staged: Staged| if staged == Staged::Unsure { Ran::Miss } else { Ran::Other };
        // The times of a call are added where it lands: a call that
        // leaves the path earlier is in none of them.
        let clock = graph::timed().then(std::time::Instant::now);
        let spent = [const { std::cell::Cell::new(0u64) }; 3];
        let lap = |at: usize, since: Option<std::time::Instant>| {
            if let Some(since) = since {
                spent[at].set(since.elapsed().as_nanos() as u64);
            }
            since.map(|_| std::time::Instant::now())
        };
        let staged = self.run_stage(site, path, Stage::Before, &mut own, function, api);
        if staged != Staged::Done {
            leave(own);
            return other(staged);
        }
        // The forms, the one that ran last first.
        let mut found = None;
        for k in 0..count {
            let Some(exported) = path.exported.borrow().get(k).cloned() else {
                break;
            };
            let fits = of_world(&exported)
                && self.accepts(&exported, &mut own, api)
                && exported.state == *state.get_or_insert_with(|| graph::thread_state());
            if fits {
                found = Some((k, exported));
                break;
            }
        }
        let Some((at, exported)) = found else {
            leave(own);
            return Ran::Other;
        };
        if at > 0 {
            let mut forms = path.exported.borrow_mut();
            if forms.get(at).is_some_and(|form| Arc::ptr_eq(form, &exported)) {
                let form = forms.remove(at);
                forms.insert(0, form);
            }
        }
        let Some(callable) = &exported.callable else {
            // Closed for numbers that were new at every call: where they
            // have been the same for a few calls in a row, the kind of call
            // is open again, and the next one is taken down.
            if !exported.varying.is_empty() {
                let now: Vec<Obj> = exported.varying.iter().map(|place| place.at(&own)).collect();
                let same = {
                    let last = exported.last.borrow();
                    last.len() == now.len() && last.iter().zip(&now).all(|(was, is)| !is.is_null() && graph::same_value(was.as_ptr(), *is))
                };
                if same {
                    if exported.repeats.fetch_add(1, Ordering::Relaxed) + 1 >= graph::SETTLED {
                        path.exported.borrow_mut().retain(|other| !Arc::ptr_eq(other, &exported));
                    }
                } else {
                    exported.repeats.store(0, Ordering::Relaxed);
                    let kept = now.iter().filter(|o| !o.is_null() && !capi::is_unboxed(**o)).map(|o| PyObj::from_borrowed(*o));
                    *exported.last.borrow_mut() = kept.collect();
                }
            }
            leave(own);
            return Ran::Miss;
        };
        // What the path lets go of where it lands (what the registers and
        // the variables of the function hold and it overwrites) must be
        // something whose end runs no Python code, or stay alive.
        if !self.lets_all_go(path, &own, regs, (frames.top(), frames.depth == 0), early) {
            // The path is the one this call takes, and cannot be run for
            // it: no other is looked for or taken down, the instructions
            // run. A graph that this happens to call after call gives way
            // to the graphs inside it.
            site.turned_down();
            if site.gave_way() {
                // The graphs of the function that were part of this one
                // have something to do again: the blocks are looked at
                // anew, for the code that enters them.
                for slot in &self.settled.blocks {
                    slot.set(Settled { code: std::ptr::null(), ..slot.get() });
                }
            }
            leave(own);
            return Ran::Miss;
        }
        let args: Vec<Obj> = exported.args.iter().map(|place| place.at(&own)).collect();
        let bc = path.body[0].bc;
        frames.top().at(bc & OFFSET_MASK);
        if self.unbound.any.get() {
            self.bind();
        }
        let expected = (path.outputs.len() + path.exits.len()) as isize;
        let result = {
            // What was made of the graph is not the function's own code.
            let _nobody = Hooked::nobody();
            let mut held = true;
            for (position, check) in &exported.guards {
                let holds = ffi::PyObject_GetAttrString(check.as_ptr(), c"holds".as_ptr());
                let tensor = args.get(*position).copied().unwrap_or(null_mut());
                let asked = !holds.is_null() && !tensor.is_null();
                let said = if asked { ffi::PyObject_CallOneArg(holds, tensor) } else { null_mut() };
                held &= !said.is_null() && ffi::PyObject_IsTrue(said) == 1;
                xdecref(holds);
                xdecref(said);
            }
            if !held {
                ffi::PyErr_Clear();
                leave(own);
                return Ran::Miss;
            }
            let clock = lap(0, clock);
            let result = ffi::PyObject_Vectorcall(callable.as_ptr(), args.as_ptr(), args.len(), null_mut());
            lap(1, clock);
            let fits = result.is_null()
                || ((ffi::PyTuple_Check(result) != 0 || ffi::PyList_Check(result) != 0)
                    && ffi::PyObject_Size(result) == expected);
            if fits {
                result
            } else {
                ffi::Py_DECREF(result);
                set_error(
                    ffi::PyExc_TypeError,
                    format!("what was made of the graph does not return a tuple of its {} results", expected),
                );
                null_mut()
            }
        };
        if result.is_null() {
            // One that has run before raises what it raises: nothing was
            // done yet, and the instructions would raise it too. One that
            // has not is taken back, and the instructions run from then on.
            if exported.proven.load(Ordering::Relaxed) || ffi::PyErr_ExceptionMatches(ffi::PyExc_Exception) == 0 {
                leave(own);
                return Ran::Landed(Landing { block: usize::MAX, code: std::ptr::null(), pc: bc as usize });
            }
            let reason = graph::raised_text();
            path.exported.borrow_mut().retain(|other| !Arc::ptr_eq(other, &exported));
            site.give_up(format!("the compiled graph failed: {}", reason));
            leave(own);
            return Ran::Miss;
        }
        exported.proven.store(true, Ordering::Relaxed);
        let clock = graph::timed().then(std::time::Instant::now);
        let item = |i: usize| {
            if ffi::PyTuple_Check(result) != 0 {
                ffi::PyTuple_GET_ITEM(result, i as isize)
            } else {
                ffi::PyList_GET_ITEM(result, i as isize)
            }
        };
        // What the guards of the code ask of the tensors that stay inside.
        let kept = path.exits.iter().enumerate().all(|(i, cond)| {
            let found = ffi::PyLong_AsLongLong(item(path.outputs.len() + i));
            match cond {
                GuardCond::TensorRank { min_rank, .. } => found >= *min_rank as i64,
                GuardCond::DimRange { lo, hi, .. } => found >= *lo && found <= *hi,
            }
        });
        if !kept {
            // The form gives what the code is not for: the instructions
            // run, this time and on every later call.
            ffi::PyErr_Clear();
            ffi::Py_DECREF(result);
            path.exported.borrow_mut().retain(|other| !Arc::ptr_eq(other, &exported));
            leave(own);
            return Ran::Miss;
        }
        for (i, v) in path.outputs.iter().enumerate() {
            capi::release(std::mem::replace(&mut own[*v as usize], new_ref(item(i))));
        }
        ffi::Py_DECREF(result);
        // Nothing of what the callable did is to be seen yet: where an
        // instruction behind it cannot run, its results are dropped.
        let staged = self.run_stage(site, path, Stage::After, &mut own, function, api);
        if staged != Staged::Done {
            leave(own);
            return other(staged);
        }
        site.sure();
        exported.hits.fetch_add(1, Ordering::Relaxed);
        // The registers of the function, as they are where the path ends.
        for (made, v) in &path.writes {
            let held = reg(&own, *made);
            capi::retain(held);
            capi::release(std::mem::replace(&mut regs[*v as usize], held));
        }
        for v in &path.clears {
            capi::release(take(regs, *v));
        }
        for (slot, made) in &path.bound {
            self.store_local(frames.top(), *slot, reg(&own, *made));
        }
        leave(own);
        if lap(2, clock).is_some() {
            for (at, took) in spent.iter().enumerate() {
                site.times[at].fetch_add(took.get(), Ordering::Relaxed);
            }
            site.times[3].fetch_add(1, Ordering::Relaxed);
        }
        let (code, pc) = path.resume.unwrap_or((std::ptr::null(), AFRESH));
        Ran::Landed(Landing { block: path.land.0, code, pc })
    }

    /// Runs a node of a path that is being recorded, as `exec_dispatch`
    /// does, and says which site ran: null if none did and the generic
    /// operation made no new one.
    unsafe fn exec_node(&self, regs: &mut [Obj], d: &DispatchOp, api: &TcfgApi) -> (OpResult, *mut capi::TcfgSite) {
        let ptrs: Vec<Obj> = d.operands.iter().map(|v| reg(regs, *v)).collect();
        let flags = if self.options.unboxed { capi::TCFG_UNBOXED_RESULT } else { 0 };
        let known = d.slot.len();
        for i in 0..known {
            let mut status = 0i32;
            let r = (api.site_call)(d.slot.get(i), ptrs.as_ptr(), flags, &mut status);
            if !r.is_null() {
                d.slot.hits.fetch_add(1, Ordering::Relaxed);
                return (Ok(r), d.slot.get(i));
            }
            if status != capi::TCFG_MISS {
                return (Err(()), null_mut());
            }
        }
        let result = self.exec_slot(regs, &d.slot, ptrs.as_ptr(), flags, &d.fallback, api);
        let learned = result.is_ok() && d.slot.len() > known;
        (result, if learned { d.slot.get(d.slot.len() - 1) } else { null_mut() })
    }

    /// The value of a path that a value of the function is: one the path
    /// made, or one it reads from before it, which is an input from then
    /// on.
    unsafe fn path_value(&self, t: &mut Trace, regs: &mut [Obj], v: V) -> PathValue {
        if v == UNDEF {
            return UNDEF;
        }
        let known = t.mapped(v);
        if known != NONE {
            return known;
        }
        if !t.outermost() {
            // A called function has nothing from before the path.
            t.lost |= !reg(regs, v).is_null();
            return UNDEF;
        }
        let kept = if reg(regs, v).is_null() { null_mut() } else { new_ref(self.obj(regs, v)) };
        t.input(v, kept)
    }

    /// How many defaults a Python function has for its last parameters.
    unsafe fn defaults_of(function: Obj) -> usize {
        let defaults = ffi::PyFunction_GetDefaults(function);
        if defaults.is_null() || ffi::PyTuple_Check(defaults) == 0 { 0 } else { ffi::PyTuple_GET_SIZE(defaults) as usize }
    }

    /// The compiled form of a function that a path goes into, which is what
    /// a path checks of every call it holds. One that is still measured
    /// against the interpreter, or was measured to run faster there and is
    /// left to it, is gone into all the same: a path does not run it in
    /// the VM, it has its instructions among its own, and reads and checks
    /// what they read at every call. None for what the frontend turned
    /// down and for what is left to the interpreter by place
    /// (`leave_to_interpreter`), of which there is no compiled form.
    unsafe fn follows(&self, function: Obj) -> Option<*const Compiled> {
        if ffi::PyFunction_Check(function) == 0 || capi::traced() {
            return None;
        }
        let form = self.function_form(function);
        if form == INTERPRETED {
            return None;
        }
        if form != DEMOTED {
            return Some(form);
        }
        let key = (ffi::PyFunction_GetCode(function) as usize, ffi::PyFunction_GetGlobals(function) as usize);
        self.callees.entries.borrow().get(&key).map(|callee| Arc::as_ptr(&callee.compiled))
    }

    /// The values that the instructions of some code of a block read from
    /// instruction `pc` on, and the blocks after it.
    fn read_from(&self, code: &BlockCode, block: usize, pc: usize) -> analysis::BitSet {
        let mut read = self.read_out(block);
        for inst in &code.insts[pc..] {
            if !matches!(inst.op, Op::Hold(_)) {
                inst.op.for_each_use(|v| {
                    read.insert(v);
                });
            }
            if let Op::Guard(g) = &inst.op {
                read.union_with(&self.read_at(block, g.resume as usize));
            }
        }
        code.term.for_each_use(|v| {
            read.insert(v);
        });
        read
    }

    /// Whether an instruction that is no call site is an operation of a
    /// graph all the same (see `graph::generic`), on the values it is about
    /// to run on: its operators, as they are reported.
    unsafe fn operation(
        &self,
        block: usize,
        origin: u32,
        inst: &Inst,
        regs: &[Obj],
        api: &TcfgApi,
        own: &dyn Fn(V) -> bool,
    ) -> Option<String> {
        let operators = self.operation_of(block, origin, inst, regs, api, own)?;
        // No operation takes a list or a tuple that holds itself: what
        // goes into the operands of an operation ends.
        let mut closed = false;
        inst.op.for_each_use(|v| closed |= graph::holds_itself(reg(regs, v)));
        (!closed).then_some(operators)
    }

    /// `operation`, before the operands are looked into.
    unsafe fn operation_of(
        &self,
        block: usize,
        origin: u32,
        inst: &Inst,
        regs: &[Obj],
        api: &TcfgApi,
        own: &dyn Fn(V) -> bool,
    ) -> Option<String> {
        let mut info = TcfgTensorInfo::empty();
        let mut tensor = |v: V| !reg(regs, v).is_null() && (api.tensor_info)(reg(regs, v), &mut info) != 0;
        // A subscript store into a tensor of the path's own.
        if let Op::SetItem { obj, .. } = &inst.op {
            let st = self.state.borrow();
            let found = st.blocks[block].facts.get(origin as usize);
            let fine = tensor(*obj) && own(*obj) && !found.is_some_and(|f| f.frames || f.raised);
            let operators = found.map_or(String::new(), |f| f.operators.join("+"));
            return fine.then(|| if operators.is_empty() { "setitem".to_string() } else { operators });
        }
        // The first tensor it is handed, in the order of the call.
        let mut first: Option<V> = None;
        let mut look = |v: V| {
            if first.is_none() && tensor(v) {
                first = Some(v);
            }
        };
        match &inst.op {
            Op::Call { args, .. } => args.iter().for_each(|v| look(*v)),
            Op::CallMethod { recv, args, .. } => std::iter::once(recv).chain(args.iter()).for_each(|v| look(*v)),
            _ => {}
        }
        let fits = match &inst.op {
            // A method that was bound to a tensor earlier has its receiver
            // in no operand.
            Op::Call { callee, .. } => {
                let c = reg(regs, *callee);
                let function = !c.is_null() && !capi::is_unboxed(c) && ffi::PyCFunction_Check(c) != 0;
                let bound = if function { ffi::PyCFunction_GetSelf(c) } else { null_mut() };
                function && (bound.is_null() || (api.tensor_info)(bound, &mut info) == 0)
            }
            Op::CallMethod { .. } => true,
            Op::GetItem(a, _) => tensor(*a),
            _ => false,
        };
        if !fits || origin == NONE {
            return None;
        }
        let st = self.state.borrow();
        let found = st.blocks[block].facts.get(origin as usize)?;
        let known = found.runs > 0
            && matches!(found.kernels, Kernels::One | Kernels::Many)
            && !(found.frames || found.raised)
            && matches!(found.callee, spec::CalleeFact::Unseen)
            && !found.operators.is_empty();
        let gives = inst.dst != NONE && matches!(st.obs.get(inst.dst as usize), Some(ValueObs::Tensor { .. }));
        let pure = found.operators.iter().all(|name| graph::pure_operator(name));
        // What writes, writes to its first tensor and to nothing else, and
        // that is the path's own; or it is handed no tensor, and writes to
        // what it makes.
        let written = |name: &String| graph::pure_operator(name) || graph::writes_first(name);
        let writes = !pure && found.operators.iter().all(written) && first.is_none_or(own);
        let fine = if writes { gives || inst.dst == NONE } else { pure && gives && !found.wrote };
        (known && fine).then(|| found.operators.join("+"))
    }

    /// Takes down one instruction of a path: decides whether it can be part
    /// of one, on the values it is about to run on, and if so runs it as
    /// the block loop does and adds it to the trace, renamed.
    #[allow(clippy::too_many_arguments)]
    unsafe fn record_inst(
        &self,
        t: &mut Trace,
        site: &Arc<GraphSite>,
        block: usize,
        code: &BlockCode,
        pc: usize,
        regs: &mut [Obj],
        frames: &mut Frames,
        api: &TcfgApi,
    ) -> Taken {
        let inst = &code.insts[pc];
        let origin = code.origins[pc];
        // A graph that starts further on in the function is part of this
        // one. One of a called function is not: it is met where the
        // function runs without a path around it.
        if t.outermost() {
            for other in code.graphs.iter().filter(|other| other.root == (block, origin as usize)) {
                if !Arc::ptr_eq(other, site) {
                    *other.within.borrow_mut() = Arc::downgrade(site);
                }
            }
        } else {
            let mut unknown = false;
            inst.op.for_each_use(|v| unknown |= v != UNDEF && t.mapped(v) == NONE && !reg(regs, v).is_null());
            if unknown {
                return Taken::Ends("a called function reads what the path does not have".to_string());
            }
        }
        let kill = |t: &mut Trace, regs: &mut [Obj]| {
            for v in &inst.kills {
                capi::release(take(regs, *v));
                t.kill(*v);
            }
        };
        // What only renames, binds or holds.
        match &inst.op {
            Op::Graph(_) => return Taken::Went,
            Op::Hold(_) => {
                kill(t, regs);
                return Taken::Went;
            }
            Op::FrameEnter(function) => {
                // The frame of a call that is inlined. A path that ends in
                // there ends in front of it: where a path ends, the frames
                // are those of the function alone.
                if frames.enter(self.func.consts[*function as usize].as_ptr()).is_err() {
                    return Taken::Raised;
                }
                if t.outermost() {
                    match &mut t.mark {
                        Some(mark) => mark.frames += 1,
                        None => t.mark((block, code as *const BlockCode, pc), 1),
                    }
                    t.inlined.push(t.bound.len());
                }
                return Taken::Went;
            }
            Op::FrameLeave => {
                // A path that began inside the call does not take its frame
                // away: it ends here.
                if t.outermost() && t.inlined.is_empty() {
                    return Taken::Ends("the end of a call that is inlined".to_string());
                }
                self.leave(frames);
                if t.outermost() {
                    if let Some(bound) = t.inlined.pop() {
                        t.bound.truncate(bound);
                    }
                    if t.mark.as_mut().is_some_and(|mark| {
                        mark.frames -= 1;
                        mark.frames == 0
                    }) {
                        t.mark = None;
                    }
                }
                return Taken::Went;
            }
            Op::Guard(g) => {
                if !Self::guard_holds(&g.cond, regs, Some(api)) {
                    return Taken::Deopt(g.resume as usize);
                }
                let cond = graph::renamed_guard(&g.cond, &mut |v| self.path_value(t, regs, v));
                t.check(Assert::Guard(cond), inst.bc);
                kill(t, regs);
                return Taken::Went;
            }
            Op::StoreLocal(slot, v) => {
                self.store_local(frames.top(), *slot, reg(regs, *v));
                // The frame of a called function is gone where the path ends.
                if t.outermost() {
                    let bound = self.path_value(t, regs, *v);
                    t.bound.push((*slot, bound));
                }
                kill(t, regs);
                return Taken::Went;
            }
            Op::LoadSlot(slot) | Op::StoreSlot(slot, _) => {
                // A variable that a handler finds: the value it is.
                let (from, to) = match &inst.op {
                    Op::StoreSlot(_, v) => (*v, *slot),
                    _ => (*slot, inst.dst),
                };
                let held = self.path_value(t, regs, from);
                let Ok(obj) = self.exec_op(regs, &inst.op) else {
                    return Taken::Raised;
                };
                if inst.dst != NONE {
                    capi::release(std::mem::replace(&mut regs[inst.dst as usize], obj));
                }
                t.rename(to, held);
                kill(t, regs);
                return Taken::Went;
            }
            _ => {}
        }
        // What the instruction is to a path, or why it is no part of one.
        let made = |t: &Trace, v: V| t.mapped(v) != NONE && t.defs[t.mapped(v) as usize].made;
        if let Some((list, item)) = self.appends(t, inst, regs).filter(|(list, _)| made(t, *list)) {
            // `list.append(item)` on a list the path made: the list gets
            // the item, and the call gives None.
            let (list, item) = (self.path_value(t, regs, list), self.path_value(t, regs, item));
            frames.top().at(inst.bc & OFFSET_MASK);
            let Ok(nothing) = self.exec_op(regs, &inst.op) else {
                return Taken::Raised;
            };
            // What is read of a list that holds what an operation gave is
            // read behind the graph, and so is what is put into it later
            // (`Trace::note`).
            t.push(Op::ListAppend(list, item), NONE, Kind::Residual, inst.bc);
            ffi::Py_XINCREF(nothing);
            let dst = t.define(inst.dst, Def::default(), nothing);
            t.push(Op::Const(self.func.const_none), dst, Kind::Pure, inst.bc);
            capi::release(std::mem::replace(&mut regs[inst.dst as usize], nothing));
            kill(t, regs);
            return Taken::Went;
        }
        if let Op::GetItem(list, at) = &inst.op {
            // An item of a list the path filled is the value that was put
            // there, for as long as the place is the same number.
            let place = reg(regs, *at);
            let items = if made(t, *list) { self.built(t, t.mapped(*list)) } else { None };
            let plain = !place.is_null() && !capi::is_unboxed(place) && ffi::PyLong_CheckExact(place) != 0;
            if let (Some(items), true) = (items, plain && ffi::PyList_CheckExact(reg(regs, *list)) != 0) {
                let count = items.len() as isize;
                let asked = ffi::PyLong_AsSsize_t(place);
                let found = if asked < 0 { asked + count } else { asked };
                if (0..count).contains(&found) && count == ffi::PyList_GET_SIZE(reg(regs, *list)) {
                    let Ok(item) = self.exec_op(regs, &inst.op) else {
                        return Taken::Raised;
                    };
                    let number = self.path_value(t, regs, *at);
                    t.check(Assert::Equals(number, PyObj::from_borrowed(place)), inst.bc);
                    capi::release(std::mem::replace(&mut regs[inst.dst as usize], item));
                    t.rename(inst.dst, items[found as usize]);
                    kill(t, regs);
                    return Taken::Went;
                }
            }
        }
        let owned = |t: &Trace, v: V| t.mapped(v) != NONE && t.defs[t.mapped(v) as usize].own;
        let operators = self.operation(block, origin, inst, regs, api, &|v| owned(t, v));
        if operators.is_none() {
            if let Some(taken) = self.record_call(t, site, (block, code as *const BlockCode, pc), regs, frames, api) {
                return taken;
            }
        }
        if let (Op::CallMethod { recv, name, args, kwnames }, None) = (&inst.op, &operators) {
            // A call of what an object has under a name, where that is a
            // function a path may call: the read, and then the call.
            let load = Op::LoadAttr(*recv, *name);
            let held = reg(regs, *recv);
            let read = *kwnames == NONE && !held.is_null() && !capi::is_unboxed(held) && self.quiet(regs, &load, api);
            let found = if read { self.exec_op(regs, &load).unwrap_or(null_mut()) } else { null_mut() };
            ffi::PyErr_Clear();
            let given: Vec<Obj> = args.iter().map(|v| reg(regs, *v)).collect();
            let own = graph::pure_call(found, &given, |i| made(t, args[i]), api);
            if let Some(own) = own {
                frames.top().at(inst.bc & OFFSET_MASK);
                let Ok(result) = self.exec_op(regs, &inst.op) else {
                    ffi::Py_DECREF(found);
                    return Taken::Raised;
                };
                let recv = self.path_value(t, regs, *recv);
                let callee = t.derive(Op::LoadAttr(recv, *name), Kind::Read, false, found, inst.bc);
                let args: Vec<PathValue> = args.iter().map(|v| self.path_value(t, regs, *v)).collect();
                ffi::Py_XINCREF(result);
                let dst = t.derive(Op::Call { callee, args, kwnames: NONE }, Kind::Pure, own, result, inst.bc);
                if inst.dst == NONE {
                    capi::release(result);
                } else {
                    capi::release(std::mem::replace(&mut regs[inst.dst as usize], result));
                    t.rename(inst.dst, dst);
                }
                kill(t, regs);
                return Taken::Went;
            }
            xdecref(found);
        }
        let mut makes = false;
        let kind = match &inst.op {
            Op::Dispatch(d) if d.slot.traits() == 0 && d.slot.len() > 0 => Some(Kind::Node),
            // A dropout that is told not to drop anything draws nothing:
            // the form is checked for that operand, as for any number.
            Op::Dispatch(d) if d.slot.traits() == capi::SITE_RANDOM && d.slot.len() > 0 => {
                const DROPOUTS: [&str; 4] =
                    ["aten::dropout", "aten::feature_dropout", "aten::alpha_dropout", "aten::feature_alpha_dropout"];
                let dropout = d.slot.operator().is_some_and(|name| DROPOUTS.contains(&name.as_str()));
                let idle = d.operands.last().is_some_and(|train| reg(regs, *train) == ffi::Py_False());
                (dropout && idle).then_some(Kind::Node)
            }
            Op::Call { .. } | Op::CallMethod { .. } | Op::GetItem(..) | Op::SetItem { .. } if operators.is_some() => {
                Some(Kind::Node)
            }
            // An operator that writes to its first tensor and to nothing
            // else, where that tensor is the path's own.
            Op::Dispatch(d) if d.slot.traits() == capi::SITE_WRITES && d.slot.len() > 0 => {
                let mut info = TcfgTensorInfo::empty();
                let mut tensor = |v: &&V| !reg(regs, **v).is_null() && (api.tensor_info)(reg(regs, **v), &mut info) != 0;
                let written = d.operands.iter().find(|v| tensor(v));
                let fine = written.is_some_and(|v| owned(t, *v))
                    && d.slot.operator().is_some_and(|name| graph::writes_first(&name));
                fine.then_some(Kind::Node)
            }
            Op::Call { callee, args, kwnames } if *kwnames == NONE => {
                let given: Vec<Obj> = args.iter().map(|v| reg(regs, *v)).collect();
                graph::pure_call(reg(regs, *callee), &given, |i| made(t, args[i]), api).map(|own| {
                    makes = own;
                    Kind::Pure
                })
            }
            Op::CallMethod { recv, name, args, kwnames } if *kwnames == NONE => {
                let given: Vec<Obj> = args.iter().map(|v| reg(regs, *v)).collect();
                graph::pure_method(reg(regs, *recv), self.func.consts[*name as usize].as_ptr(), &given, api).then_some(Kind::Pure)
            }
            Op::BuildList(_) | Op::BuildMap(_) if self.quiet(regs, &inst.op, api) => {
                makes = true;
                Some(Kind::Pure)
            }
            Op::ListAppend(target, _)
            | Op::MapAdd(target, ..)
            | Op::IterNext(target)
            | Op::ListExtend(target, _)
            | Op::DictMerge(target, ..) => (made(t, *target) && self.quiet(regs, &inst.op, api)).then_some(Kind::Residual),
            Op::GetIter(v) => match graph::quiet_iter(reg(regs, *v)) {
                Some(of_path) if !of_path || made(t, *v) => {
                    makes = true;
                    Some(Kind::Residual)
                }
                _ => None,
            },
            Op::LoadGlobal(_) | Op::LoadDeref(_) | Op::CellGet(..) => Some(Kind::Read),
            Op::LoadAttr(..)
            | Op::GetItem(..)
            | Op::Contains(..)
            | Op::GetLen(_)
            | Op::Unpack(..)
            | Op::ImportName { .. } => self.quiet(regs, &inst.op, api).then_some(Kind::Read),
            op => self.quiet(regs, op, api).then_some(Kind::Pure),
        };
        let ends = || {
            let st = self.state.borrow();
            let bs = &st.blocks[block];
            let found = bs.facts.get(origin as usize);
            let reason = bs.reasons.get(origin as usize).and_then(|r| r.as_ref());
            Taken::Ends(graph::why_not(&self.func, &inst.op, found, reason))
        };
        let Some(kind) = kind else {
            return ends();
        };
        let Some(mut renamed) = graph::renamed(&inst.op, &mut |v| self.path_value(t, regs, v)) else {
            return ends();
        };
        if let (Op::LoadDeref(cell), false) = (&inst.op, t.outermost()) {
            // The closure of a called function is that of the function the
            // call was found to enter.
            renamed = Op::Part(Part::Cell(*cell), t.function());
        }
        if kind == Kind::Node {
            // A list the path made and filled is handed over as what it
            // holds: the graph has those values, not the list.
            let mut lists: Vec<(PathValue, PathValue)> = Vec::new();
            graph::reads(&renamed, |v| {
                // One that was also extended is taken item by item, as it
                // is by now.
                let extended = t.body.iter().any(|inst| matches!(&inst.op, Op::ListExtend(target, _) if *target == v));
                let held = t.rec.values[v as usize];
                let items = if extended && !held.is_null() && ffi::PyList_CheckExact(held) != 0 {
                    let count = ffi::PyList_GET_SIZE(held) as usize;
                    t.check(Assert::Sequence(v, ffi::Py_TYPE(held) as usize, count), inst.bc);
                    let item = |t: &mut Trace, i: usize| {
                        let kept = new_ref(ffi::PyList_GET_ITEM(held, i as isize));
                        t.derive(Op::Part(Part::Index(i as u32), v), Kind::Pure, false, kept, inst.bc)
                    };
                    Some((0..count).map(|i| item(t, i)).collect())
                } else {
                    self.listed(t, v)
                };
                if let Some(items) = items {
                    let late = items.iter().any(|item| t.defs[*item as usize].late);
                    let op = Op::BuildList(items);
                    let graph = late && graph::computed(&op, &t.defs, |_| false);
                    let kept = t.rec.values[v as usize];
                    ffi::Py_XINCREF(kept);
                    let list = t.value(Def { node: false, late, graph, made: false, own: false }, kept);
                    t.source[list as usize] = t.source[v as usize];
                    t.push(op, list, Kind::Pure, inst.bc);
                    lists.push((v, list));
                }
            });
            if !lists.is_empty() {
                let again = graph::renamed(&renamed, &mut |v| lists.iter().find(|(l, _)| *l == v).map_or(v, |(_, to)| *to));
                renamed = again.unwrap_or(renamed);
            }
            // A node takes nothing that was computed from a node and that
            // the graph cannot compute itself.
            let mut fits = true;
            graph::reads(&renamed, |v| {
                let taken = t.defs[v as usize];
                fits &= !taken.late || taken.node || taken.graph;
            });
            if !fits {
                return Taken::Ends("an operation takes what the graph cannot compute".to_string());
            }
        }
        // It runs as the block loop runs it.
        frames.top().at(inst.bc & OFFSET_MASK);
        if self.unbound.any.get() && inst.op.is_call() && !matches!(inst.op, Op::Dispatch(_)) {
            self.bind();
        }
        let result = match (&inst.op, kind) {
            (Op::Dispatch(d), Kind::Node) => {
                // A write to what is the path's own is no write to what it
                // was given.
                let given: Vec<Obj> = d.operands.iter().filter(|v| !owned(t, **v)).map(|v| reg(regs, *v)).collect();
                let versions = versions_of(&given, api);
                let (result, picked) = self.exec_node(regs, d, api);
                t.rec.wrote |= versions_moved(&versions, api);
                if result.is_ok() && picked.is_null() && t.rec.failed.is_none() {
                    t.rec.failed = Some("an operation of the graph ran without a call site".to_string());
                }
                result.map(|obj| (obj, picked))
            }
            (op, Kind::Node) => {
                let mut given: Vec<Obj> = Vec::new();
                op.for_each_use(|v| {
                    if !owned(t, v) {
                        given.push(reg(regs, v));
                    }
                });
                let versions = versions_of(&flattened(&given), api);
                let result = self.exec_op(regs, op);
                t.rec.wrote |= versions_moved(&versions, api);
                result.map(|obj| (obj, null_mut()))
            }
            (op, _) => self.exec_op(regs, op).map(|obj| (obj, null_mut())),
        };
        let result = match result {
            Ok((obj, _)) if inst.op.is_call() && ffi::PyErr_CheckSignals() != 0 => {
                capi::release(obj);
                Err(())
            }
            other => other,
        };
        let Ok((obj, picked)) = result else {
            return Taken::Raised;
        };
        // What it gave is kept for the graph to be described with: a tensor
        // with its Python object.
        let obj = if capi::is_unboxed(obj) {
            let boxed = (api.tensor_box)(obj);
            if boxed.is_null() {
                panic!("torch.vm: could not create the Python object of a tensor");
            }
            boxed
        } else {
            obj
        };
        // What a node gave: a tensor, or nothing that is read (a store, a
        // write whose result is dropped).
        let mut info = TcfgTensorInfo::empty();
        let gave = !obj.is_null() && (api.tensor_info)(obj, &mut info) != 0;
        let silent = inst.dst == NONE && (matches!(inst.op, Op::SetItem { .. }) || gave);
        if kind == Kind::Node {
            let writes = match &inst.op {
                Op::Dispatch(d) => d.slot.traits() & capi::SITE_WRITES != 0,
                Op::SetItem { .. } => true,
                _ => operators.as_ref().is_some_and(|names| names.split('+').any(|name| !graph::pure_operator(name))),
            };
            t.inplace |= writes;
        }
        if kind == Kind::Node && !gave && !silent && t.rec.failed.is_none() {
            // Several tensors, or none (`split`, `max` along a dimension,
            // `meshgrid`): it has run, and is no operation of a graph. The
            // path ends in front of it.
            return Taken::Cut(obj, "an operation gives no single tensor".to_string());
        }
        let mut dst = NONE;
        if inst.dst != NONE {
            let mut late = kind == Kind::Node;
            graph::reads(&renamed, |v| late |= t.defs[v as usize].late);
            let int = |v: PathValue| {
                let held = t.rec.values[v as usize];
                !held.is_null() && ffi::PyLong_CheckExact(held) != 0
            };
            let node = kind == Kind::Node;
            let graph = late && !node && graph::computed(&renamed, &t.defs, int);
            ffi::Py_XINCREF(obj);
            // Its memory is that of an operand it gave back or is a view
            // of, and then as much the path's own as that is, or new.
            let mut own = node && gave && !info.data.is_null();
            if own {
                let memory = info.data;
                let mut other = TcfgTensorInfo::empty();
                graph::reads(&renamed, |v| {
                    let held = t.rec.values[v as usize];
                    if held.is_null() {
                        return;
                    }
                    if (api.tensor_info)(held, &mut other) != 0 {
                        own &= other.data != memory || t.defs[v as usize].own;
                    } else if ffi::PyList_CheckExact(held) != 0 || ffi::PyTuple_CheckExact(held) != 0 {
                        // What a list holds is not known to be the path's.
                        for item in flattened(&[held]) {
                            own &= (api.tensor_info)(item, &mut other) == 0 || other.data != memory;
                        }
                    }
                });
            }
            dst = t.define(inst.dst, Def { node, late, graph, made: makes, own }, obj);
            capi::release(std::mem::replace(&mut regs[inst.dst as usize], obj));
        } else {
            capi::release(obj);
        }
        if kind == Kind::Node {
            let mut info = TcfgTensorInfo::empty();
            let held = if dst == NONE { null_mut() } else { t.rec.values[dst as usize] };
            let tensor = !held.is_null() && (api.tensor_info)(held, &mut info) != 0;
            if !tensor && !silent && t.rec.failed.is_none() {
                t.rec.failed = Some("an operation of the graph gave no tensor".to_string());
            }
            // Its type is read from its Python object, which a mode would
            // see: only for a run that is described.
            let described = tensor && t.rec.failed.is_none();
            t.rec.nodes.push((picked, if described { graph::tensor_meta(held) } else { null_mut() }));
            t.operators.push(match &inst.op {
                Op::Dispatch(d) => d.slot.operator().unwrap_or_else(|| "?".to_string()),
                _ => operators.unwrap_or_else(|| "?".to_string()),
            });
            if t.operators.len() == 1 && t.outermost() && origin != NONE {
                t.first = Some((block, origin));
            }
        }
        t.push(renamed, dst, kind, inst.bc);
        kill(t, regs);
        Taken::Went
    }

    /// Takes down, in place of an instruction that enters a Python
    /// function, the instructions of that function (`follow`): a call, a
    /// call of a method, and what the interpreter does through a function
    /// of the class of an object (iterating over it, counting it, taking an
    /// item of it, reading an attribute that is a property, a descriptor
    /// of a class defined in Python, or left to `__getattr__`). None for
    /// any other instruction, which is then looked at as it was before.
    unsafe fn record_call(
        &self,
        t: &mut Trace,
        site: &Arc<GraphSite>,
        at: (usize, *const BlockCode, usize),
        regs: &mut [Obj],
        frames: &mut Frames,
        api: &TcfgApi,
    ) -> Option<Taken> {
        let inst = &(&(*at.1).insts)[at.2];
        let names = graph::specials();
        let f = &self.func;
        let special = |v: V, name: &PyObj| !graph::special(reg(regs, v), name).is_null();
        let form = match &inst.op {
            Op::Call { callee, args, kwnames } => {
                let held = reg(regs, *callee);
                let plain = *kwnames == NONE && !held.is_null() && !capi::is_unboxed(held);
                // A builtin that asks its arguments for an iterator, one of
                // them through an `__iter__` written in Python, and the
                // others in C.
                let over = if plain {
                    let given: Vec<Obj> = args.iter().map(|v| reg(regs, *v)).collect();
                    let asked = &args[..graph::goes_over(held, &given)];
                    let python = |v: &V| special(*v, &names.iter);
                    let goes = asked.iter().any(python) && asked.iter().all(|v| python(v) || graph::quiet_iter(reg(regs, *v)).is_some());
                    if goes { asked.len() } else { 0 }
                } else {
                    0
                };
                if plain && args.len() == 1 && graph::is_len(held) && special(args[0], &names.len) {
                    Form::Special(&names.len, args[0], None)
                } else if over > 0 {
                    return Some(self.record_over(t, site, at, *callee, args, over, regs, frames, api));
                } else {
                    Form::Call { callee: *callee, args, kwnames: *kwnames }
                }
            }
            Op::CallMethod { recv, name, args, kwnames } => {
                Form::Method { recv: *recv, name: *name, args, kwnames: *kwnames }
            }
            Op::CallEx { callee, args, kwargs } => Form::Spread { callee: *callee, args: *args, kwargs: *kwargs },
            Op::GetIter(v) if special(*v, &names.iter) => Form::Special(&names.iter, *v, None),
            Op::GetItem(a, key) if special(*a, &names.getitem) => Form::Special(&names.getitem, *a, Some(*key)),
            Op::LoadAttr(a, name) if !reg(regs, *a).is_null() && !self.quiet(regs, &inst.op, api) => {
                let descr = graph::data_descriptor(reg(regs, *a), f.consts[*name as usize].as_ptr());
                if descr.is_null() {
                    Form::Getattr(*a, *name)
                } else if ffi::Py_TYPE(descr) == std::ptr::addr_of_mut!(ffi::PyProperty_Type) {
                    Form::Property(*a, *name)
                } else {
                    Form::Descriptor(*a, *name)
                }
            }
            _ => return None,
        };
        let (result, value) = match self.follow(t, site, at, regs, frames, api, &form) {
            Followed::Not => return None,
            Followed::Raised => return Some(Taken::Raised),
            // The function ran to its end off the path, which ends in front
            // of the instruction: that gives what the interpreter makes of
            // what the function returned or raised.
            Followed::Left(result, reason) => return Some(Taken::Cut(self.makes_of(&form, inst, regs, result), reason)),
            Followed::Returned(result, value) => (result, value),
        };
        // What the interpreter makes of what such a function returns. A path
        // goes on where that is the value itself under a check the path can
        // make at every call (an iterator, a count); anything else
        // is made of it here (`makes_of`), and the path ends in front.
        if let Form::Special(name, ..) = &form {
            let (goes, check, why) = if std::ptr::eq(*name, &names.iter) {
                (ffi::PyIter_Check(result) != 0, Assert::Iterator(value), "an __iter__ returns what is no iterator")
            } else if std::ptr::eq(*name, &names.len) {
                (graph::is_count(result), Assert::Count(value), "a __len__ returns what is no count")
            } else {
                (true, Assert::Count(NONE), "")
            };
            if !goes {
                return Some(Taken::Cut(self.makes_of(&form, inst, regs, result), why.to_string()));
            }
            if !why.is_empty() {
                t.check(check, inst.bc);
            }
        }
        t.unmark();
        if inst.dst == NONE {
            capi::release(result);
        } else {
            capi::release(std::mem::replace(&mut regs[inst.dst as usize], result));
            t.rename(inst.dst, value);
        }
        for v in &inst.kills {
            capi::release(take(regs, *v));
            t.kill(*v);
        }
        Some(Taken::Went)
    }

    /// What an instruction gives for what the Python function it went into
    /// on its behalf returned (null: the function raised, and the error is
    /// set), as the interpreter makes it of that. Every way out of such a
    /// function on which the path does not go on comes through here:
    /// `iter` and `for` raise for what is no iterator; `len` gives the
    /// index of what `__len__` returned as an int and raises for a negative
    /// one; a read of an attribute whose property or descriptor raised
    /// AttributeError asks the `__getattr__` of the class, if it has one.
    /// Any other instruction gives what the function returned, and raises
    /// what it raised. Takes the reference to `result`; gives one, or null
    /// with the error set.
    unsafe fn makes_of(&self, form: &Form, inst: &Inst, regs: &mut [Obj], result: Obj) -> Obj {
        let names = graph::specials();
        match (form, &inst.op) {
            (Form::Property(v, name) | Form::Descriptor(v, name), _) if result.is_null() => {
                self.asks_getattr(reg(regs, *v), self.func.consts[*name as usize].as_ptr())
            }
            _ if result.is_null() => null_mut(),
            (Form::Special(name, ..), _) if std::ptr::eq(*name, &names.iter) => {
                if ffi::PyIter_Check(result) == 0 {
                    set_error(ffi::PyExc_TypeError, format!("iter() returned non-iterator of type '{}'", pyffi::type_name(result)));
                    ffi::Py_DECREF(result);
                    return null_mut();
                }
                result
            }
            (Form::Special(name, ..), _) if std::ptr::eq(*name, &names.len) => counted(result),
            _ => result,
        }
    }

    /// For a read of an attribute that raised AttributeError in the lookup
    /// of `object` (the getter of a property, the `__get__` of a
    /// descriptor): what the `__getattr__` of the class gives, if the class
    /// has one, which the interpreter then asks (`slot_tp_getattr_hook`).
    /// Null with the error as it is for any other error and for a class
    /// without one.
    unsafe fn asks_getattr(&self, object: Obj, name: Obj) -> Obj {
        if object.is_null() || capi::is_unboxed(object) || ffi::PyErr_ExceptionMatches(ffi::PyExc_AttributeError) == 0 {
            return null_mut();
        }
        let class = ffi::Py_TYPE(object);
        let hook = pyffi::_PyType_Lookup(class, graph::specials().getattr.as_ptr());
        if hook.is_null() {
            return null_mut();
        }
        ffi::PyErr_Clear();
        let _kept = PyObj::from_borrowed(hook);
        let bound = match (*ffi::Py_TYPE(hook)).tp_descr_get {
            Some(get) => get(hook, object, class as Obj),
            None => new_ref(hook),
        };
        if bound.is_null() {
            return null_mut();
        }
        let given = ffi::PyObject_CallOneArg(bound, name);
        ffi::Py_DECREF(bound);
        given
    }

    /// `record_call` for a builtin that asks its arguments for an iterator
    /// (`graph::goes_over`) where the class of one has an `__iter__` in
    /// Python: the path goes into that function. Where what it returned is
    /// an iterator that C code goes over and the path came back from the
    /// function, the builtin is called with that iterator and the path goes
    /// on. On every other way out the builtin is called with what stands
    /// for the object (`iterated`), which gives it what the `__iter__`
    /// returned, and the path ends in front of the call.
    #[allow(clippy::too_many_arguments)]
    unsafe fn record_over(
        &self,
        t: &mut Trace,
        site: &Arc<GraphSite>,
        at: (usize, *const BlockCode, usize),
        callee: V,
        args: &[V],
        over: usize,
        regs: &mut [Obj],
        frames: &mut Frames,
        api: &TcfgApi,
    ) -> Taken {
        let inst = &(&(*at.1).insts)[at.2];
        let names = graph::specials();
        let builtin = reg(regs, callee);
        // What the builtin is handed where the path goes on: for an
        // argument whose `__iter__` the path went into, the iterator that
        // returned, which is one that C code goes over.
        let mut given: Vec<PyObj> = args.iter().map(|v| PyObj::from_borrowed(self.obj(regs, *v))).collect();
        // And where it cannot: what stands for such an argument
        // (`iterated`), so that the builtin does with what the `__iter__`
        // returned what it does when it calls the `__iter__` itself.
        let mut stands = given.clone();
        let mut values: Vec<PathValue> = vec![NONE; args.len()];
        let call = |handed: &[PyObj]| {
            let objects: Vec<Obj> = handed.iter().map(|o| o.as_ptr()).collect();
            ffi::PyObject_Vectorcall(builtin, objects.as_ptr(), objects.len(), null_mut())
        };
        let mut went = false;
        for (i, v) in args.iter().enumerate().take(over) {
            if graph::special(reg(regs, *v), &names.iter).is_null() {
                continue;
            }
            let form = Form::Special(&names.iter, *v, None);
            let (returned, value, stopped) = match self.follow(t, site, at, regs, frames, api, &form) {
                Followed::Returned(returned, value) => (returned, value, "an __iter__ returns what is no iterator".to_string()),
                Followed::Raised => return Taken::Raised,
                Followed::Left(result, reason) if result.is_null() => return Taken::Cut(result, reason),
                Followed::Left(result, reason) => (result, NONE, reason),
                // No `__iter__` has been called: the call is the
                // interpreter's.
                Followed::Not if !went => {
                    return Taken::Ends(format!("{} goes over what enters Python code", pyffi::function_str(builtin)));
                }
                Followed::Not => return Taken::Cut(call(&stands), "an __iter__ that is not gone into".to_string()),
            };
            went = true;
            let returned = PyObj::from_owned(returned);
            let stand = iterated(returned.as_ptr());
            if stand.is_null() {
                return Taken::Cut(stand, stopped);
            }
            stands[i] = PyObj::from_owned(stand);
            if value == NONE || ffi::PyIter_Check(returned.as_ptr()) == 0 {
                return Taken::Cut(call(&stands), stopped);
            }
            t.check(Assert::Iterator(value), inst.bc);
            given[i] = returned;
            values[i] = value;
        }
        let objects: Vec<Obj> = given.iter().map(|o| o.as_ptr()).collect();
        let Some(own) = graph::pure_call(builtin, &objects, |i| values[i] != NONE, api) else {
            return Taken::Cut(call(&stands), "a call that goes over what enters Python code".to_string());
        };
        let result = call(&given);
        if result.is_null() {
            return Taken::Raised;
        }
        if graph::makes_sequence(builtin) {
            // `list` and `tuple` ask an object that has a `__len__` or a
            // `__length_hint__` for it: this one's class has neither.
            let object = self.path_value(t, regs, args[0]);
            t.check(Assert::Lacks(object, names.len.clone()), inst.bc);
            t.check(Assert::Lacks(object, graph::length_hint()), inst.bc);
        }
        let function = self.path_value(t, regs, callee);
        for (i, v) in args.iter().enumerate() {
            if values[i] == NONE {
                values[i] = self.path_value(t, regs, *v);
            }
        }
        let made = t.derive(Op::Call { callee: function, args: values, kwnames: NONE }, Kind::Pure, own, new_ref(result), inst.bc);
        t.unmark();
        if inst.dst == NONE {
            capi::release(result);
        } else {
            capi::release(std::mem::replace(&mut regs[inst.dst as usize], result));
            t.rename(inst.dst, made);
        }
        for v in &inst.kills {
            capi::release(take(regs, *v));
            t.kill(*v);
        }
        Taken::Went
    }

    /// Goes into the Python function that an instruction enters, if that
    /// function runs compiled, is not being measured against the
    /// interpreter and is not already on the trace: runs it as `enter`
    /// does, with its instructions taken down and its parameters being
    /// the values the instruction hands it. The path checks, where it is
    /// run, that what is called still comes down to a function that runs
    /// as the same compiled function (`Assert::Enters`), and reads the
    /// receiver, the defaults and the closure from what is called then.
    #[allow(clippy::too_many_arguments)]
    unsafe fn follow(
        &self,
        t: &mut Trace,
        site: &Arc<GraphSite>,
        at: (usize, *const BlockCode, usize),
        regs: &mut [Obj],
        frames: &mut Frames,
        api: &TcfgApi,
        form: &Form,
    ) -> Followed {
        let f = &self.func;
        let bc = (&(*at.1).insts)[at.2].bc;
        let names = graph::specials();
        let not = |why: &str| {
            if graph::debug() {
                eprintln!("torch.vm graph {}: {} is not gone into: {}", site.name, form.text(), why);
            }
            Followed::Not
        };
        if capi::traced() || !capi::has_frames() || t.acts.len() >= MAX_PATH_DEPTH {
            return not("frames are traced, or the path is too deep");
        }
        // What is called, what it comes down to, and what it is handed.
        let mut read: Vec<PyObj> = Vec::new();
        let (mut called, mut receiver, mut descr): (Obj, Obj, Obj) = (null_mut(), null_mut(), null_mut());
        let mut items: Vec<Obj> = Vec::new();
        // The names of the values that are handed over under one, which
        // are the last of them.
        let mut keys: Vec<Obj> = Vec::new();
        let func = match form {
            Form::Call { callee, args, kwnames: kw } | Form::Method { recv: callee, args, kwnames: kw, .. } => {
                let held = reg(regs, *callee);
                if held.is_null() || capi::is_unboxed(held) {
                    return Followed::Not;
                }
                called = held;
                if let Form::Method { recv, name, .. } = form {
                    let load = Op::LoadAttr(*recv, *name);
                    if !self.quiet(regs, &load, api) {
                        return Followed::Not;
                    }
                    let Ok(found) = self.exec_op(regs, &load) else {
                        ffi::PyErr_Clear();
                        return Followed::Not;
                    };
                    read.push(PyObj::from_owned(found));
                    called = found;
                }
                let Some((func, bound, _)) = self.target(called) else {
                    return Followed::Not;
                };
                receiver = bound;
                if !receiver.is_null() {
                    items.push(receiver);
                }
                for v in args.iter() {
                    items.push(self.obj(regs, *v));
                }
                if *kw != NONE {
                    let names = f.consts[*kw as usize].as_ptr();
                    keys.extend((0..ffi::PyTuple_GET_SIZE(names)).map(|k| ffi::PyTuple_GET_ITEM(names, k)));
                }
                func
            }
            Form::Spread { callee, args, kwargs } => {
                called = reg(regs, *callee);
                let given = reg(regs, *args);
                let plain = |o: Obj| !o.is_null() && !capi::is_unboxed(o);
                let sequence = plain(given) && (ffi::PyTuple_CheckExact(given) != 0 || ffi::PyList_CheckExact(given) != 0);
                let Some((func, bound, _)) = (if plain(called) && sequence { self.target(called) } else { None }) else {
                    return Followed::Not;
                };
                receiver = bound;
                if !receiver.is_null() {
                    items.push(receiver);
                }
                items.extend((0..ffi::PyObject_Size(given)).map(|i| graph::sequence_item(given, i)));
                if *kwargs != NONE {
                    let named = reg(regs, *kwargs);
                    if !plain(named) || ffi::PyDict_CheckExact(named) == 0 {
                        return Followed::Not;
                    }
                    let (mut at, mut key, mut value) = (0, null_mut(), null_mut());
                    while ffi::PyDict_Next(named, &mut at, &mut key, &mut value) != 0 {
                        if ffi::PyUnicode_CheckExact(key) == 0 {
                            return Followed::Not;
                        }
                        keys.push(key);
                        items.push(value);
                    }
                }
                func
            }
            Form::Special(name, v, key) => {
                items.push(reg(regs, *v));
                if let Some(key) = key {
                    items.push(self.obj(regs, *key));
                }
                graph::special(items[0], name)
            }
            Form::Property(v, name) => {
                descr = graph::data_descriptor(reg(regs, *v), f.consts[*name as usize].as_ptr());
                let getter = ffi::PyObject_GetAttrString(descr, c"fget".as_ptr());
                if getter.is_null() {
                    ffi::PyErr_Clear();
                    return Followed::Not;
                }
                read.push(PyObj::from_owned(getter));
                items.push(reg(regs, *v));
                getter
            }
            Form::Descriptor(v, name) => {
                descr = graph::data_descriptor(reg(regs, *v), f.consts[*name as usize].as_ptr());
                let held = reg(regs, *v);
                items.extend([descr, held, ffi::Py_TYPE(held) as Obj]);
                graph::special(descr, &names.get)
            }
            Form::Getattr(v, name) => {
                let attr = f.consts[*name as usize].as_ptr();
                if !graph::absent(reg(regs, *v), attr) {
                    return Followed::Not;
                }
                items.extend([reg(regs, *v), attr]);
                graph::special(items[0], &names.getattr)
            }
        };
        if func.is_null() || ffi::PyFunction_Check(func) == 0 {
            return not("it comes down to no Python function");
        }
        let Some(compiled) = self.follows(func) else {
            return not("the function is left to the interpreter");
        };
        let c = &*compiled;
        let here = std::ptr::addr_of!(compiled) as usize;
        // A function that the trace is in may be gone into again (a
        // `Sequential` inside a `Sequential`), a few times.
        let again = t.acts.iter().filter(|act| t.owners[act.owner as usize] == compiled).count();
        if again >= MAX_PATH_AGAIN {
            return not("the path is in that function too many times already");
        }
        if here < STACK_FLOOR.with(|floor| *floor) + STACK_MARGIN {
            return not("the native stack is nearly used up");
        }
        // The arguments are bound as `enter` binds them.
        let cf = &c.func;
        let nkw = keys.len();
        let total = items.len() - nkw;
        let kwargs = if nkw == 0 { null_mut() } else { ffi::PyDict_New() };
        for k in 0..nkw {
            ffi::PyDict_SetItem(kwargs, keys[k], items[total + k]);
        }
        let nregs = cf.nvalues as usize + 1;
        let mut inner = c.frames.borrow_mut().pop().unwrap_or_else(|| vec![null_mut(); nregs]);
        inner[nregs - 1] = new_ref(func);
        let done = |inner: Vec<Obj>| {
            let mut inner = inner;
            for r in inner.iter_mut() {
                capi::release(std::mem::replace(r, null_mut()));
            }
            c.frames.borrow_mut().push(inner);
        };
        let positional = Args { receiver: null_mut(), items: items.as_ptr(), n: total };
        let fits = c.bind_args(&mut inner, func, &positional, kwargs).is_ok();
        xdecref(kwargs);
        let rest = cf.argcount + cf.kwonlyargcount;
        if !fits {
            // The call says what does not fit where the instructions make it.
            ffi::PyErr_Clear();
            done(inner);
            return not("the arguments do not fit");
        }
        // The same binding, of the values of the path.
        if t.outermost() && t.mark.is_none() {
            t.mark(at, 0);
        }
        let mut given: Vec<PathValue> = Vec::new();
        let function = match form {
            Form::Call { callee, args, .. } | Form::Method { recv: callee, args, .. } => {
                let mut value = self.path_value(t, regs, *callee);
                if let Form::Method { name, .. } = form {
                    value = t.derive(Op::LoadAttr(value, *name), Kind::Read, false, new_ref(called), bc);
                }
                let function = t.derive(Op::Part(Part::Target, value), Kind::Read, false, new_ref(func), bc);
                if receiver == called {
                    given.push(value);
                } else if !receiver.is_null() {
                    given.push(t.derive(Op::Part(Part::Receiver, value), Kind::Read, false, new_ref(receiver), bc));
                }
                for v in args.iter() {
                    given.push(self.path_value(t, regs, *v));
                }
                function
            }
            Form::Spread { callee, args, kwargs } => {
                let value = self.path_value(t, regs, *callee);
                let function = t.derive(Op::Part(Part::Target, value), Kind::Read, false, new_ref(func), bc);
                if receiver == called {
                    given.push(value);
                } else if !receiver.is_null() {
                    given.push(t.derive(Op::Part(Part::Receiver, value), Kind::Read, false, new_ref(receiver), bc));
                }
                // What the path built itself is taken apart as it was
                // built; anything else is checked to be made alike.
                let first = given.len();
                let sequence = self.path_value(t, regs, *args);
                match self.built(t, sequence).filter(|built| built.len() == total - first) {
                    Some(built) => given.extend(built),
                    None => {
                        let class = ffi::Py_TYPE(reg(regs, *args)) as usize;
                        t.check(Assert::Sequence(sequence, class, total - first), bc);
                        for i in first..total {
                            let part = Op::Part(Part::Index((i - first) as u32), sequence);
                            given.push(t.derive(part, Kind::Pure, false, new_ref(items[i]), bc));
                        }
                    }
                }
                if *kwargs != NONE {
                    let named = self.path_value(t, regs, *kwargs);
                    let pinned = self.pins_keys(t, named, bc);
                    let entries = self.entries(t, named).filter(|entries| {
                        pinned && entries.len() == nkw && entries.iter().zip(&keys).all(|((key, _), name)| key == name)
                    });
                    match entries {
                        Some(entries) => given.extend(entries.iter().map(|(_, value)| *value)),
                        None => {
                            let names = keys.iter().map(|key| PyObj::from_borrowed(*key)).collect();
                            t.check(Assert::Keys(named, names), bc);
                            for k in 0..nkw {
                                let part = Op::Part(Part::Entry(PyObj::from_borrowed(keys[k])), named);
                                given.push(t.derive(part, Kind::Pure, false, new_ref(items[total + k]), bc));
                            }
                        }
                    }
                }
                function
            }
            Form::Special(name, v, key) => {
                given.push(self.path_value(t, regs, *v));
                if let Some(key) = key {
                    given.push(self.path_value(t, regs, *key));
                }
                t.derive(Op::Part(Part::OfType((*name).clone()), given[0]), Kind::Read, false, new_ref(func), bc)
            }
            Form::Property(v, name) | Form::Descriptor(v, name) => {
                let value = self.path_value(t, regs, *v);
                let part = Part::Descriptor(f.consts[*name as usize].clone());
                let found = t.derive(Op::Part(part, value), Kind::Read, false, new_ref(descr), bc);
                if let Form::Property(..) = form {
                    given.push(value);
                    t.derive(Op::Part(Part::Getter, found), Kind::Read, false, new_ref(func), bc)
                } else {
                    let class = t.derive(Op::Part(Part::Type, value), Kind::Read, false, new_ref(items[2]), bc);
                    given.extend([found, value, class]);
                    t.derive(Op::Part(Part::OfType(names.get.clone()), found), Kind::Read, false, new_ref(func), bc)
                }
            }
            Form::Getattr(v, name) => {
                let value = self.path_value(t, regs, *v);
                t.check(Assert::Absent(value, f.consts[*name as usize].clone()), bc);
                let attr = t.derive(Op::Const(*name), Kind::Pure, false, new_ref(items[1]), bc);
                given.extend([value, attr]);
                t.derive(Op::Part(Part::OfType(names.getattr.clone()), value), Kind::Read, false, new_ref(func), bc)
            }
        };
        t.check(Assert::Enters(function, compiled, Self::defaults_of(func)), bc);
        t.enter(compiled, cf.nvalues as usize, function);
        for i in 0..total.min(cf.argcount) {
            t.rename(cf.params[i], given[i]);
        }
        let kept = |v: V| new_ref(reg(&inner, v));
        if cf.has_varargs {
            let more = given[cf.argcount.min(total)..total].to_vec();
            let tuple = t.derive(Op::BuildTuple(more), Kind::Pure, false, kept(cf.params[rest]), bc);
            t.rename(cf.params[rest], tuple);
        }
        // A name the function has no parameter for goes into its dict of
        // those, in the order of the call.
        let mut spare: Vec<PathValue> = Vec::new();
        for k in 0..nkw {
            let name = keys[k];
            let named = (cf.posonlyargcount..rest).find(|j| {
                let param = cf.param_names[*j].as_ptr();
                param == name || ffi::PyObject_RichCompareBool(param, name, ffi::Py_EQ) > 0
            });
            match named {
                Some(j) => t.rename(cf.params[j], given[total + k]),
                None => {
                    let key = t.derive(Op::Object(PyObj::from_borrowed(name)), Kind::Pure, false, new_ref(name), bc);
                    spare.extend([key, given[total + k]]);
                }
            }
        }
        if cf.has_varkw {
            let v = cf.params[rest + cf.has_varargs as usize];
            let dict = t.derive(Op::BuildMap(spare), Kind::Pure, true, kept(v), bc);
            t.rename(v, dict);
        }
        let defaults = ffi::PyFunction_GetDefaults(func);
        let ndefaults = if defaults.is_null() { 0 } else { ffi::PyTuple_GET_SIZE(defaults) as usize };
        for j in 0..rest {
            let v = cf.params[j];
            if t.mapped(v) != NONE || reg(&inner, v).is_null() {
                continue;
            }
            let part = if j < cf.argcount {
                Part::Default((j + ndefaults - cf.argcount) as u32)
            } else {
                Part::KwDefault(cf.param_names[j].clone())
            };
            let default = t.derive(Op::Part(part, function), Kind::Read, false, kept(v), bc);
            t.rename(v, default);
        }
        // The function runs in a frame of its own, as `enter` and `run`
        // have it, with its instructions taken down.
        frames.top().at(bc & OFFSET_MASK);
        if self.unbound.any.get() {
            self.bind();
        }
        let Ok(frame) = Frame::push(func) else {
            done(inner);
            return Followed::Raised;
        };
        if !frame.locals.is_null() {
            for (slot, v) in cf.params.iter().enumerate() {
                let o = reg(&inner, *v);
                if !o.is_null() {
                    *frame.locals.add(slot) = frame_ref(new_ref(o));
                }
            }
        }
        let mut below = Frames::new(frame);
        let walked = c.walk(t, site, &mut inner, &mut below, api, 0, None);
        // Where the trace does not come back, the function runs to its end
        // without it.
        let (result, back) = match walked {
            Walked::Returned(result, value) => (result, Ok(value)),
            Walked::Ended { block, code, pc, afresh, reason } => {
                t.provisional |= afresh;
                let landing = Landing { block, code, pc: if afresh { AFRESH } else { pc } };
                (c.exec_blocks(&mut inner, &mut below, Some(PathEnd::Landed(landing))), Err(reason))
            }
            Walked::Left { block, code, call, reason } => {
                let landing = Landing { block, code, pc: call + 1 };
                (c.exec_blocks(&mut inner, &mut below, Some(PathEnd::Landed(landing))), Err(reason))
            }
            Walked::Raised(block, bc) => {
                let raised = Some(PathEnd::Raised(block, bc));
                (c.exec_blocks(&mut inner, &mut below, raised), Err("it raised".to_string()))
            }
            Walked::Deopt(block, at) => {
                let deopt = Some(PathEnd::Deopt(block, at));
                (c.exec_blocks(&mut inner, &mut below, deopt), Err("a guard of its code failed".to_string()))
            }
        };
        while below.depth > 0 {
            c.leave(&mut below);
        }
        // What the call held ends with it, also where the trace keeps it.
        let mut suspects: Vec<Obj> = inner.iter().copied().filter(|held| !held.is_null() && !capi::is_unboxed(*held)).collect();
        if !frame.locals.is_null() {
            let variables = (0..cf.params.len()).map(|slot| frame_owned(*frame.locals.add(slot)));
            suspects.extend(variables.filter(|held| !held.is_null() && !capi::is_unboxed(*held)));
        }
        let sweep = Self::ends_for_good(t, &suspects);
        let result = c.returned(frame, result);
        done(inner);
        if self.overdue(t, &mut suspects, sweep) {
            let reason = t.abandoned.clone().unwrap_or_default();
            return if result.is_null() { Followed::Raised } else { Followed::Left(result, format!("in {}: {}", cf.name, reason)) };
        }
        match back {
            Ok(value) if !result.is_null() => {
                t.leave();
                Followed::Returned(result, value)
            }
            Ok(_) => Followed::Raised,
            Err(reason) => Followed::Left(result, format!("in {}: {}", cf.name, reason)),
        }
    }

    /// For `list.append(item)` on an exact list, however the code has it:
    /// the list and the item.
    unsafe fn appends(&self, t: &Trace, inst: &Inst, regs: &[Obj]) -> Option<(V, V)> {
        let (list, name, args, held) = self.method_of(t, inst, regs)?;
        let named = pyffi::str_to_string(self.func.consts[name as usize].as_ptr()) == "append";
        let exact = !held.is_null() && !capi::is_unboxed(held) && ffi::PyList_CheckExact(held) != 0;
        (named && exact && args.len() == 1 && inst.dst != NONE).then(|| (list, args[0]))
    }

    /// A call of a method without names, as the two ways a function has
    /// of writing it: the register the method is called on, its name,
    /// what it is handed, and the value it is called on. `a.append(item)`
    /// is one instruction; `add = a.append` and `add(item)` later are a
    /// read of the method and a call of what was read. For the second
    /// the method is one written in C that has exactly the value it was
    /// read off, which the path has taken down (the register may be
    /// empty by now): a method that an object merely holds,
    /// `o.append = other.append`, is a method of the other value, and no
    /// method of the object.
    unsafe fn method_of<'a>(&self, t: &Trace, inst: &'a Inst, regs: &[Obj]) -> Option<(V, u32, &'a [V], Obj)> {
        match &inst.op {
            Op::CallMethod { recv, name, args, kwnames: NONE } => Some((*recv, *name, args.as_slice(), reg(regs, *recv))),
            Op::Call { callee, args, kwnames: NONE } => {
                let bound = reg(regs, *callee);
                if bound.is_null() || capi::is_unboxed(bound) || ffi::PyCFunction_Check(bound) == 0 {
                    return None;
                }
                let Some(Op::LoadAttr(recv, name)) = self.analysis.def_op(&self.func, *callee) else {
                    return None;
                };
                let (held, read_off) = (ffi::PyCFunction_GetSelf(bound), t.mapped(*recv));
                let same = read_off != NONE && !held.is_null() && t.rec.values[read_off as usize] == held;
                same.then(|| (*recv, *name, args.as_slice(), held))
            }
            _ => None,
        }
    }

    /// For a tuple or a list that a path built: the values it holds where
    /// the trace has got to.
    fn built(&self, t: &Trace, sequence: PathValue) -> Option<Vec<PathValue>> {
        Self::built_by(t, sequence, t.body.len())
    }

    /// `built` as the first `upto` instructions of the trace left it. A
    /// list that was spread into another (`[*a, b]`, `tuple(a)`) gave it
    /// the items it held where that instruction stands, so what it held
    /// is asked as of there: every step from one sequence to another goes
    /// on in front of the instruction it comes from, and so this ends,
    /// whatever the path put into what.
    fn built_by(t: &Trace, sequence: PathValue, upto: usize) -> Option<Vec<PathValue>> {
        let mut items: Option<Vec<PathValue>> = None;
        for (at, inst) in t.body[..upto].iter().enumerate() {
            match &inst.op {
                Op::BuildTuple(vs) | Op::BuildList(vs) if inst.dst == sequence => items = Some(vs.clone()),
                Op::ListToTuple(list) if inst.dst == sequence => items = Self::built_by(t, *list, at),
                Op::ListAppend(target, v) if *target == sequence => items.as_mut()?.push(*v),
                Op::ListExtend(target, more) if *target == sequence => items.as_mut()?.extend(Self::built_by(t, *more, at)?),
                _ => {}
            }
        }
        items
    }

    /// For a dict that a path built: its keys, as they were in this run,
    /// and the values it holds under them where the trace has got to.
    fn entries(&self, t: &Trace, dict: PathValue) -> Option<Vec<(Obj, PathValue)>> {
        Self::entries_by(t, dict, t.body.len())
    }

    /// `entries` as the first `upto` instructions of the trace left it:
    /// a dict that was merged into another gave it the entries it had
    /// where that instruction stands (see `built_by`, also for why this
    /// ends).
    fn entries_by(t: &Trace, dict: PathValue, upto: usize) -> Option<Vec<(Obj, PathValue)>> {
        let mut entries: Option<Vec<(Obj, PathValue)>> = None;
        let key = |v: PathValue| t.rec.values[v as usize];
        for (at, inst) in t.body[..upto].iter().enumerate() {
            match &inst.op {
                Op::BuildMap(vs) if inst.dst == dict => {
                    entries = Some(vs.chunks(2).map(|pair| (key(pair[0]), pair[1])).collect());
                }
                Op::MapAdd(target, k, v) if *target == dict => {
                    let entries = entries.as_mut()?;
                    match entries.iter_mut().find(|(other, _)| *other == key(*k)) {
                        Some(entry) => entry.1 = *v,
                        None => entries.push((key(*k), *v)),
                    }
                }
                Op::DictMerge(target, more, _) if *target == dict => entries.as_mut()?.extend(Self::entries_by(t, *more, at)?),
                _ => {}
            }
        }
        entries
    }

    /// Whether the keys under which a path put entries into a dict it made,
    /// and took them out, are the same at every call: each is a constant
    /// or is made one here by a check (`Assert::Equals`). What `entries`
    /// says of the dict holds at another call only with this: with other
    /// keys the same instructions leave other entries. False where a key
    /// is computed behind the callable or cannot be checked, and for a
    /// dict that took the entries of one of which this is not so.
    unsafe fn pins_keys(&self, t: &mut Trace, dict: PathValue, bc: u32) -> bool {
        if dict == NONE || dict == UNDEF {
            return false;
        }
        let (mut keys, mut merged): (Vec<PathValue>, Vec<(usize, PathValue)>) = (Vec::new(), Vec::new());
        for (at, inst) in t.body.iter().enumerate() {
            match &inst.op {
                Op::BuildMap(vs) if inst.dst == dict => keys.extend(vs.iter().step_by(2)),
                Op::MapAdd(target, key, _) if *target == dict => keys.push(*key),
                Op::CallMethod { recv, args, .. } if *recv == dict && inst.kind == Kind::Residual => keys.extend(args.first()),
                Op::DictMerge(target, more, _) if *target == dict => merged.push((at, *more)),
                _ => {}
            }
        }
        // The dict that was merged was put together in front of the merge.
        for (at, more) in merged {
            let built = t.body[..at].iter().any(|inst| inst.dst == more && matches!(inst.op, Op::BuildMap(_)));
            if !built || !self.pins_keys(t, more, bc) {
                return false;
            }
        }
        for key in keys {
            let (of, held) = (t.defs[key as usize], t.rec.values[key as usize]);
            // A constant of its function is one wherever the path took
            // it from (a path that starts behind it has it as an input).
            let (owner, v) = t.source[key as usize];
            let constant = v != NONE && v != UNDEF && {
                let function = &*t.owners[owner as usize];
                matches!(function.analysis.def_op(&function.func, v), Some(Op::Const(_)))
            };
            // A constant that the path loads, or pinned by a check already.
            let known = |inst: &graph::PathInst| match &inst.assert {
                Some(Assert::Equals(v, _)) => *v == key,
                _ => inst.dst == key && matches!(inst.op, Op::Const(_)),
            };
            if constant || t.body.iter().any(known) {
                continue;
            }
            if of.late || held.is_null() || !graph::checkable(held) {
                return false;
            }
            t.check(Assert::Equals(key, graph::frozen(held)), bc);
        }
        true
    }

    /// For a list that a path made: the values it holds where the trace has
    /// got to, if the path filled it by nothing but appending. None for any
    /// other value, and for a list that is as its instruction built it.
    fn listed(&self, t: &Trace, list: PathValue) -> Option<Vec<PathValue>> {
        let mut items: Option<Vec<PathValue>> = None;
        let mut appended = false;
        for inst in &t.body {
            match &inst.op {
                Op::BuildList(vs) if inst.dst == list => items = Some(vs.clone()),
                Op::ListAppend(target, v) if *target == list => {
                    items.as_mut()?.push(*v);
                    appended = true;
                }
                _ => {}
            }
        }
        items.filter(|_| appended)
    }

    /// Takes the instructions of the function down as they run, as the
    /// block loop would run them, through the blocks as this call goes
    /// through them: from instruction `pc` of some code of `block`
    /// (`from`), or from the start of the block in the code that is
    /// selected for it. Up to the first instruction that can be no part of
    /// a path, or, in a called function, to where it returns.
    #[allow(clippy::too_many_arguments)]
    unsafe fn walk(
        &self,
        t: &mut Trace,
        site: &Arc<GraphSite>,
        regs: &mut [Obj],
        frames: &mut Frames,
        api: &TcfgApi,
        block: usize,
        from: Option<(*const BlockCode, usize)>,
    ) -> Walked {
        let func = &self.func;
        let mut b = block;
        let (mut code, mut pc) = match from {
            Some(from) => from,
            None => {
                let (found, mode) = self.select(b, regs, Some(api));
                if mode == Mode::Profile {
                    return Walked::Ended { block: b, code: found, pc: 0, afresh: true, reason: NOT_PROFILED.to_string() };
                }
                (found, 0)
            }
        };
        // The guards a version was entered under hold for the values the
        // path starts with, whichever version enters it later.
        let guards = |t: &mut Trace, regs: &mut [Obj], code: &BlockCode, bc: u32| {
            for cond in &code.entry_guards {
                if !reg(regs, cond.value()).is_null() {
                    let cond = graph::renamed_guard(cond, &mut |v| self.path_value(t, regs, v));
                    t.check(Assert::Guard(cond), bc);
                }
            }
        };
        let ended = |block: usize, code: *const BlockCode, pc: usize, reason: &str| {
            Walked::Ended { block, code, pc, afresh: false, reason: reason.to_string() }
        };
        guards(t, regs, &*code, (&(*code).insts).get(pc).map_or(func.blocks[b].term_bc, |inst| inst.bc));
        t.shape.extend([b as u32, (*code).id, pc as u32]);
        let mut suspects: Vec<Obj> = Vec::new();
        loop {
            let c = &*code;
            while pc < c.insts.len() {
                if t.body.len() >= MAX_PATH_STEPS {
                    return ended(b, code, pc, "the path is as long as a path may be");
                }
                // What the function lets go of here must end here, also
                // where the trace keeps it.
                let sweep = self.lets_go(t, &c.insts[pc], regs, frames.top(), &mut suspects);
                let taken = self.record_inst(t, site, b, c, pc, regs, frames, api);
                if matches!(taken, Taken::Went) && self.overdue(t, &mut suspects, sweep) {
                    let reason = t.abandoned.clone().unwrap_or_default();
                    return Walked::Left { block: b, code, call: pc, reason };
                }
                match taken {
                    Taken::Went => pc += 1,
                    Taken::Ends(reason) => return Walked::Ended { block: b, code, pc, afresh: false, reason },
                    Taken::Raised => return Walked::Raised(b, c.insts[pc].bc),
                    Taken::Deopt(at) => return Walked::Deopt(b, at),
                    Taken::Cut(result, reason) => {
                        // The call has run: the function has what it gave.
                        let inst = &c.insts[pc];
                        if result.is_null() {
                            return Walked::Raised(b, inst.bc);
                        }
                        if inst.dst == NONE {
                            capi::release(result);
                        } else {
                            capi::release(std::mem::replace(&mut regs[inst.dst as usize], result));
                        }
                        for v in &inst.kills {
                            capi::release(take(regs, *v));
                        }
                        self.overdue(t, &mut suspects, sweep);
                        return Walked::Left { block: b, code, call: pc, reason };
                    }
                }
            }
            let bc = func.blocks[b].term_bc;
            let next = match &c.term {
                Term::Jump(next) => *next,
                Term::Br { cond, then_, else_ } => {
                    let held = reg(regs, *cond);
                    if held != ffi::Py_True() && held != ffi::Py_False() {
                        return ended(b, code, pc, "a branch on what is no bool");
                    }
                    let way = held == ffi::Py_True();
                    let cond = self.path_value(t, regs, *cond);
                    t.check(Assert::Truth(cond, way), bc);
                    if way { *then_ } else { *else_ }
                }
                Term::BrNull { v, null, nonnull } => {
                    let exhausted = reg(regs, *v).is_null();
                    let v = self.path_value(t, regs, *v);
                    t.check(Assert::Null(v, exhausted), bc);
                    if exhausted { *null } else { *nonnull }
                }
                Term::Return(_) if t.outermost() => return ended(b, code, pc, "the function returns"),
                Term::Return(v) => {
                    let value = self.path_value(t, regs, *v);
                    if t.lost || reg(regs, *v).is_null() {
                        return ended(b, code, pc, "a called function returns what the path does not have");
                    }
                    self.obj(regs, *v);
                    return Walked::Returned(take(regs, *v), value);
                }
                Term::Raise { .. } | Term::Reraise(..) => return ended(b, code, pc, "the block raises"),
            } as usize;
            if t.lost {
                return ended(b, code, pc, "a called function branches on what the path does not have");
            }
            // The edge, as the block loop takes it.
            let pi = func.blocks[next].preds.iter().position(|p| *p as usize == b).unwrap_or(0);
            let moves = &self.analysis.edge_moves[next][pi];
            let cleared = c.term_kills.iter().chain(&self.analysis.edge_kills[next][pi]).chain(moves.iter().map(|(dst, _)| dst));
            suspects.extend(cleared.map(|v| reg(regs, *v)).filter(|held| !held.is_null() && !capi::is_unboxed(*held)));
            let sweep = Self::ends_for_good(t, &suspects);
            for v in &c.term_kills {
                capi::release(take(regs, *v));
                t.kill(*v);
            }
            let moved: Vec<(Obj, PathValue)> = moves
                .iter()
                .map(|(_, src)| {
                    let held = reg(regs, *src);
                    capi::retain(held);
                    (held, self.path_value(t, regs, *src))
                })
                .collect();
            for ((dst, _), (held, value)) in moves.iter().zip(moved) {
                capi::release(std::mem::replace(&mut regs[*dst as usize], held));
                t.rename(*dst, value);
            }
            for v in &self.analysis.edge_kills[next][pi] {
                capi::release(take(regs, *v));
                t.kill(*v);
            }
            b = next;
            let (found, mode) = self.select(b, regs, Some(api));
            if self.overdue(t, &mut suspects, sweep) {
                let reason = t.abandoned.clone().unwrap_or_default();
                return Walked::Ended { block: b, code: found, pc: 0, afresh: true, reason };
            }
            if mode == Mode::Profile {
                return Walked::Ended { block: b, code: found, pc: 0, afresh: true, reason: NOT_PROFILED.to_string() };
            }
            code = found;
            pc = 0;
            if t.lost {
                return ended(b, code, pc, "a called function reads what the path does not have");
            }
            guards(t, regs, &*code, func.blocks[b].bc_start);
            t.shape.extend([b as u32, (*code).id]);
        }
    }

    /// Takes a path down: runs the instructions of the function from
    /// instruction `pc` of `code`, the first operation of a graph, and
    /// those of the functions it calls (`walk`), up to the first
    /// instruction that can be no part of a path, and keeps what an
    /// exported form is made from. A path that would end inside a called
    /// function ends in front of the call. The function goes on where
    /// this run of it has got to. With a pass over graphs, the path is
    /// handed to it; a run that cannot be exported gets a form without a
    /// callable, so that the calls like it are not taken down again.
    #[allow(clippy::too_many_arguments)]
    unsafe fn record(
        &self,
        site: &Arc<GraphSite>,
        regs: &mut [Obj],
        frames: &mut Frames,
        api: &TcfgApi,
        block: usize,
        code: *const BlockCode,
        pc: usize,
        world: (u64, u64, u64, u64),
    ) -> PathEnd {
        let func = &self.func;
        // Neither does a call in which a path is taken down.
        unsettled();
        let mut t = Trace::new(self as *const Compiled, func.nvalues as usize);
        // Under a mode, or a transform, the operations run as they are: the
        // run is only looked at for what kind of call it is.
        let mode = (world.0 >> 16 != 0).then_some("a torch function mode is on");
        t.rec.failed = mode.or_else(|| graph::interposed()).map(|why| why.to_string());
        let root = (block, (&(*code).origins)[pc..].iter().find(|o| **o != NONE).map_or(0, |o| *o as usize));
        // Where this run of the function has got to.
        let mut left = false;
        let (went, ends) = match self.walk(&mut t, site, regs, frames, api, block, Some((code, pc))) {
            Walked::Ended { block, code, pc, afresh, reason } => {
                (Landing { block, code, pc: if afresh { AFRESH } else { pc } }, reason)
            }
            Walked::Left { block, code, call, reason } => {
                left = true;
                (Landing { block, code, pc: call + 1 }, reason)
            }
            Walked::Raised(block, bc) => return PathEnd::Raised(block, bc),
            Walked::Deopt(block, at) => return PathEnd::Deopt(block, at),
            Walked::Returned(..) => unreachable!("the function a path starts in is not left by it"),
        };
        if t.abandoned.is_some() {
            // The function let go of an object whose end may run Python
            // code, which the trace kept: nothing is made of this run.
            // The call was one the graph could not take, as a call is
            // that a path is turned down for.
            site.turned_down();
            if site.gave_way() {
                for slot in &self.settled.blocks {
                    slot.set(Settled { code: std::ptr::null(), ..slot.get() });
                }
            }
            return PathEnd::Landed(went);
        }
        // Where the path ends: there, or in front of the call that the
        // trace is still in, or that is inlined around where it is.
        let cut = t.mark.as_ref().map(|mark| (mark.at, mark.frames > 0));
        let (b, code, pc, afresh, ends) = match cut {
            Some(((block, code, pc), inlined)) => {
                t.back();
                (block, code, pc, false, if inlined { format!("in a call that is inlined: {}", ends) } else { ends })
            }
            // An instruction of the function itself that has run and is no
            // part of the path: the path ends in front of it.
            None if left => (went.block, went.code, went.pc - 1, false, ends),
            None => (went.block, went.code, if went.pc == AFRESH { 0 } else { went.pc }, went.pc == AFRESH, ends),
        };
        // A call that was not gone into because the function it enters is
        // still measured against the interpreter is gone into later.
        let afresh_path = afresh || t.provisional;
        t.shape.extend([u32::MAX, b as u32, if afresh { u32::MAX } else { pc as u32 }, t.body.len() as u32]);
        let c = &*code;
        let (land, resume, live) = if afresh {
            ((b, 0), None, self.without_holds().1.live_in[b].clone())
        } else {
            let index = c.origins[pc..].iter().find(|o| **o != NONE).map_or(func.blocks[b].insts.len(), |o| *o as usize);
            ((b, index), Some((code, pc)), self.read_from(c, b, pc))
        };
        // The variables that handlers find are read without an instruction
        // saying so: what the path stored in one is there when it ends.
        let mut live = live;
        for slot in &func.slots {
            live.insert(*slot);
        }
        let landing = went;
        let path = Path::of(&mut t, root, land, resume, &live, ends, afresh_path);
        if graph::debug() {
            let (steps, ops) = (path.body.len(), path.operators.len());
            eprintln!(
                "torch.vm graph {}: a path of {} instructions, {} operations ({} in front of what stands for them, {} behind; {} registers, {} results), ends at {:?}: {}",
                site.name,
                steps,
                ops,
                path.stages[0].len(),
                path.stages[1].len(),
                path.nregs,
                path.outputs.len(),
                land,
                path.ends
            );
            if std::env::var_os("TORCH_VM_GRAPH_DEBUG").is_some_and(|v| v == "2") {
                for inst in path.staged(Stage::Before).chain(path.staged(Stage::After)) {
                    let owner = &*path.owners[inst.owner as usize];
                    let what = if inst.assert.is_some() { "check" } else { "run" };
                    eprintln!("    {:?} {} in {}: %{} = {}", inst.stage, what, owner.func.qualname, inst.dst, owner.func.op_repr(&inst.op));
                }
            }
        }
        let finished = path.provisional || path.operators.len() < site.min_ops || site.exporter.is_none();
        if finished {
            site.install(path, None);
            return PathEnd::Landed(landing);
        }
        let mut described = path.describe(&site.name, &t.rec, api);
        let mut changing: Vec<graph::Place> = Vec::new();
        if described.closed.is_none() {
            // One more form for nothing but another number, with none of
            // the `EVER_NEW` before it ever run again: the numbers are new
            // every time, and what is checked for them never comes.
            if let Some(varying) = site.ever_new(&path.shape, world, &described.checks, &described.numbers) {
                changing = varying.iter().map(|k| described.checks[*k].target).collect();
                let mut at = 0;
                described.checks.retain(|_| {
                    at += 1;
                    !varying.contains(&(at - 1))
                });
                described.closed = Some(graph::EVER_NEW_REASON.to_string());
            }
        }
        let graph::Described { description, mut closed, checks, numbers, args } = described;
        if closed.is_none() && t.inplace {
            // What autograd kept of an operation was written to in place:
            // eager raises when it goes backward, and so must this call.
            // The same where that could not be found out.
            closed = self.saved_changed(&path, &t.rec);
        }
        if let (true, Some(why)) = (graph::debug(), &closed) {
            eprintln!("torch.vm graph {}: not exported for this call: {}", site.name, why);
        }
        let (modes, included, excluded, state) = world;
        let form = |callable: Option<PyObj>, guards: Vec<(usize, PyObj)>, closed: Option<String>| {
            let proven = std::sync::atomic::AtomicBool::new(false);
            let hits = AtomicU64::new(0);
            let sizes = crate::shapes::ShapeGuard::exact(&numbers);
            let varying = if closed.as_deref() == Some(graph::EVER_NEW_REASON) { changing.clone() } else { Vec::new() };
            let (last, repeats) = (crate::pyffi::Shared::new(Vec::new()), std::sync::atomic::AtomicU32::new(0));
            let form = Exported {
                modes, included, excluded, state, checks, sizes, args, guards, callable, varying, last, repeats, closed, proven, hits,
            };
            Arc::new(form)
        };
        let Some(description) = description.filter(|_| closed.is_none()) else {
            site.install(path, Some(form(None, Vec::new(), closed)));
            return PathEnd::Landed(landing);
        };
        if self.unbound.any.get() {
            self.bind();
        }
        // The time this takes says nothing about the function.
        unsettled();
        site.exports.fetch_add(1, Ordering::Relaxed);
        let made = {
            let _nobody = Hooked::nobody();
            site.export(description.as_ptr())
        };
        match made {
            Ok((callable, guards)) => {
                site.install(path, Some(form(Some(callable), guards, None)));
                plan_changed();
            }
            Err(_) if !ffi::PyErr_Occurred().is_null() => {
                let bc = c.insts.get(pc).map_or(func.blocks[b].term_bc, |inst| inst.bc);
                return PathEnd::Raised(b, bc);
            }
            Err(reason) => {
                if graph::debug() {
                    eprintln!("torch.vm graph {}: given up: {}", site.name, reason);
                }
                site.install(path, None);
                site.give_up(reason)
            }
        }
        PathEnd::Landed(landing)
    }

    // ------------------------------------------------------------------
    // The interpreter loop
    // ------------------------------------------------------------------

    unsafe fn exec(&self, regs: &mut [Obj], frame: Frame) -> Obj {
        let mut frames = Frames::new(frame);
        let result = self.exec_blocks(regs, &mut frames, None);
        // Every path out of an inlined function pops its frame; this is for
        // the one that was not thought of.
        while frames.depth > 0 {
            self.leave(&mut frames);
        }
        result
    }

    /// Runs the function: from its first block, or from where a path that
    /// was taken down through this call of it left off (`start`).
    unsafe fn exec_blocks(&self, regs: &mut [Obj], frames: &mut Frames, start: Option<PathEnd>) -> Obj {
        let api = capi::api();
        let func = &self.func;
        if self.options.specialize {
            self.settled.calls.set(self.settled.calls.get() + 1);
        }
        let serial = self.settled.calls.get();
        let mut cur: usize = 0;
        let mut pred: usize = usize::MAX;
        let mut back_edges: u32 = 0;
        let mut start = start;
        let mut warming = Warming(None);
        // An exception either continues in the handler of the block it was
        // raised in, entered like the first block, or leaves the function.
        macro_rules! raised {
            ($blocks:lifetime, $block:expr) => {
                match self.catch(regs, $block) {
                    Some(handler) => {
                        pred = usize::MAX;
                        cur = handler;
                        continue $blocks;
                    }
                    None => return null_mut(),
                }
            };
        }
        'blocks: loop {
            if pred != usize::MAX {
                let block = func.blocks.get_unchecked(cur);
                let mut pi = 0;
                while *block.preds.get_unchecked(pi) as usize != pred {
                    pi += 1;
                }
                let moves = self.analysis.edge_moves.get_unchecked(cur).get_unchecked(pi);
                if !moves.is_empty() {
                    // Phis are parallel copies: read every source first.
                    let mut small: [Obj; 8] = [null_mut(); 8];
                    let mut big: Vec<Obj>;
                    let vals: &mut [Obj] = if moves.len() <= small.len() {
                        &mut small[..moves.len()]
                    } else {
                        big = vec![null_mut(); moves.len()];
                        &mut big[..]
                    };
                    for (i, (_, src)) in moves.iter().enumerate() {
                        let o = reg(regs, *src);
                        capi::retain(o);
                        vals[i] = o;
                    }
                    for (i, (dst, _)) in moves.iter().enumerate() {
                        let old = std::mem::replace(regs.get_unchecked_mut(*dst as usize), vals[i]);
                        capi::release(old);
                    }
                }
                for v in self.analysis.edge_kills.get_unchecked(cur).get_unchecked(pi) {
                    capi::release(take(regs, *v));
                }
                if cur <= pred {
                    back_edges = back_edges.wrapping_add(1);
                    if back_edges & 63 == 0 {
                        // A signal handler or a finalizer can run here.
                        if self.unbound.any.get() {
                            self.bind();
                        }
                        if !tick() {
                            self.fail(frames, block.bc_start);
                            raised!('blocks, cur);
                        }
                    }
                }
            }

            let mut begin: Option<(*const BlockCode, Mode, usize)> = None;
            if let Some(end) = start.take() {
                match end {
                    PathEnd::Miss => {}
                    PathEnd::Raised(block, bc) => {
                        self.fail(frames, bc);
                        raised!('blocks, block);
                    }
                    PathEnd::Landed(landing) => {
                        cur = landing.block;
                        if landing.pc != AFRESH {
                            let generic = Arc::as_ptr(self.generic.get_unchecked(cur));
                            let code = if landing.code.is_null() { generic } else { landing.code };
                            let mode = if code == generic { Mode::Generic } else { Mode::Spec };
                            begin = Some((code, mode, landing.pc));
                        }
                    }
                    PathEnd::Deopt(block, at) => {
                        cur = block;
                        let mut st = self.state.borrow_mut();
                        st.blocks[cur].deopts += 1;
                        let mode = if st.blocks[cur].respec_left > 0 { Mode::Profile } else { Mode::Generic };
                        begin = Some((Arc::as_ptr(self.generic.get_unchecked(cur)), mode, at));
                    }
                }
            }
            let (mut code, mut mode, mut pc) = match begin {
                Some(begin) => begin,
                None => {
                    let (code, mode) = self.select(cur, regs, api);
                    (code, mode, 0)
                }
            };
            let _profiling = Profiling::enter(mode == Mode::Profile);
            if mode == Mode::Profile && cur == 0 {
                warming.begin(self);
            }
            let next: usize = loop {
                let mut resume: Option<usize> = None;
                let mut leap: Option<Landing> = None;
                {
                    let code_ref: &BlockCode = &*code;
                    let insts = &code_ref.insts;
                    while pc < insts.len() {
                        let inst = insts.get_unchecked(pc);
                        let result = match &inst.op {
                            Op::Guard(g) => {
                                if Self::guard_holds(&g.cond, regs, api) {
                                    Ok(null_mut())
                                } else {
                                    resume = Some(g.resume as usize);
                                    break;
                                }
                            }
                            Op::Graph(site) => {
                                let api = api.unwrap_unchecked();
                                let end = self.enter_graph(site, regs, frames, api, cur, code, pc + 1, serial);
                                if site.pending() {
                                    warming.begin(self);
                                }
                                match end {
                                    PathEnd::Miss => Ok(null_mut()),
                                    // What was made of a graph raised.
                                    PathEnd::Landed(landing) if landing.block == usize::MAX => {
                                        self.fail(frames, landing.pc as u32);
                                        raised!('blocks, cur);
                                    }
                                    PathEnd::Landed(landing) => {
                                        leap = Some(landing);
                                        break;
                                    }
                                    PathEnd::Raised(block, bc) => {
                                        self.fail(frames, bc);
                                        raised!('blocks, block);
                                    }
                                    PathEnd::Deopt(block, at) => {
                                        cur = block;
                                        resume = Some(at);
                                        break;
                                    }
                                }
                            }
                            Op::CallObject(..) => {
                                let frame = frames.top();
                                frame.at(inst.bc & OFFSET_MASK);
                                if self.unbound.any.get() {
                                    self.bind();
                                }
                                let outer = REGION_FRAME.with(|f| f.replace(frame));
                                let result = {
                                    let _nobody = Hooked::nobody();
                                    self.exec_op(regs, &inst.op)
                                };
                                REGION_FRAME.with(|f| f.set(outer));
                                result
                            }
                            Op::StoreLocal(slot, v) => {
                                self.store_local(frames.top(), *slot, reg(regs, *v));
                                Ok(null_mut())
                            }
                            Op::FrameEnter(function) => {
                                let function = func.consts.get_unchecked(*function as usize).as_ptr();
                                frames.enter(function).map(|()| null_mut())
                            }
                            Op::FrameLeave => {
                                self.leave(frames);
                                Ok(null_mut())
                            }
                            Op::CaughtLasti => Ok(frames.top().lasti()),
                            op => {
                                frames.top().at(inst.bc & OFFSET_MASK);
                                // What is called can look at the frames.
                                if self.unbound.any.get() && op.is_call() && !matches!(op, Op::Dispatch(_)) {
                                    self.bind();
                                }
                                if mode == Mode::Profile {
                                    self.exec_profiled(cur, pc, inst, regs, api.unwrap())
                                } else {
                                    self.exec_op(regs, op)
                                }
                            }
                        };
                        // The interpreter looks for pending signals when a
                        // call returns; a handler runs there, and what it
                        // raises is raised by the call.
                        let result = match result {
                            Ok(obj) if inst.op.is_call() && ffi::PyErr_CheckSignals() != 0 => {
                                capi::release(obj);
                                Err(())
                            }
                            other => other,
                        };
                        match result {
                            Ok(obj) => {
                                if inst.dst != NONE {
                                    let old = std::mem::replace(
                                        regs.get_unchecked_mut(inst.dst as usize),
                                        obj,
                                    );
                                    capi::release(old);
                                }
                                for v in &inst.kills {
                                    capi::release(take(regs, *v));
                                }
                            }
                            Err(()) => {
                                self.fail(frames, inst.bc);
                                if mode == Mode::Profile {
                                    self.raised_profiling(cur);
                                }
                                raised!('blocks, cur);
                            }
                        }
                        pc += 1;
                    }
                    if resume.is_none() && leap.is_none() {
                        if mode == Mode::Profile {
                            self.specialize(cur);
                        }
                        // What a frame that was left reports as its line.
                        let term_bc = func.blocks.get_unchecked(cur).term_bc;
                        frames.top().at(term_bc & OFFSET_MASK);
                        let next = match &code_ref.term {
                            Term::Jump(b) => *b as usize,
                            Term::Br { cond, then_, else_ } => {
                                let c = self.obj(regs, *cond);
                                let truth = if c == ffi::Py_True() {
                                    true
                                } else if c == ffi::Py_False() {
                                    false
                                } else {
                                    let r = ffi::PyObject_IsTrue(c);
                                    if r < 0 {
                                        self.fail(frames, term_bc);
                                        raised!('blocks, cur);
                                    }
                                    r != 0
                                };
                                (if truth { *then_ } else { *else_ }) as usize
                            }
                            Term::BrNull { v, null, nonnull } => {
                                (if reg(regs, *v).is_null() { *null } else { *nonnull }) as usize
                            }
                            Term::Return(v) => {
                                // What leaves the VM is a Python object.
                                self.obj(regs, *v);
                                return take(regs, *v);
                            }
                            Term::Raise { exc, cause } => {
                                let bc = term_bc;
                                let handled =
                                    if *exc == NONE { pyffi::PyErr_GetHandledException() } else { null_mut() };
                                if !handled.is_null() && handled != ffi::Py_None() {
                                    // A bare raise: the exception being handled
                                    // goes on as it is.
                                    ffi::PyErr_SetRaisedException(handled);
                                } else if *exc == NONE {
                                    xdecref(handled);
                                    set_error(
                                        ffi::PyExc_RuntimeError,
                                        "No active exception to reraise".to_string(),
                                    );
                                    self.fail(frames, bc);
                                } else {
                                    let cause =
                                        if *cause == NONE { null_mut() } else { self.obj(regs, *cause) };
                                    do_raise(self.obj(regs, *exc), cause);
                                    self.fail(frames, bc);
                                }
                                raised!('blocks, cur);
                            }
                            Term::Reraise(exc, lasti) => {
                                if *lasti != NONE {
                                    frames.top().restore(reg(regs, *lasti));
                                }
                                ffi::PyErr_SetRaisedException(new_ref(self.obj(regs, *exc)));
                                raised!('blocks, cur);
                            }
                        };
                        for v in &code_ref.term_kills {
                            capi::release(take(regs, *v));
                        }
                        break next;
                    }
                }
                if let Some(landing) = leap {
                    // The registers are what they are at that instruction:
                    // the function goes on from there, in the code that is
                    // named, or in the generic tier.
                    cur = landing.block;
                    if landing.pc == AFRESH {
                        pred = usize::MAX;
                        continue 'blocks;
                    }
                    let generic = Arc::as_ptr(self.generic.get_unchecked(cur));
                    code = if landing.code.is_null() { generic } else { landing.code };
                    mode = if code == generic { Mode::Generic } else { Mode::Spec };
                    pc = landing.pc;
                    continue;
                }
                // A guard failed: this block goes back to the generic tier,
                // from the instruction the guard names.
                let mut st = self.state.borrow_mut();
                st.blocks[cur].deopts += 1;
                mode = if st.blocks[cur].respec_left > 0 { Mode::Profile } else { Mode::Generic };
                drop(st);
                code = Arc::as_ptr(self.generic.get_unchecked(cur));
                pc = resume.unwrap();
            };
            pred = cur;
            cur = next;
        }
    }
}
