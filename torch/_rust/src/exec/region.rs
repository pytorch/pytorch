// Regions: a pure block's version compiled as one graph.
//
// Once a pure block's (unguarded beyond the optimizer's own checks) version
// has run often enough, its next run is traced: torch._rust_inductor's
// trace_region runs the body (through _region_body) under make_fx on the
// real inputs, and hands the aten graph to a backend (Inductor). The block
// then gets a version in front of the others whose guard adds what the trace
// depended on (each tensor's dtype, device, sizes, strides and requires_grad,
// each scalar's value, each object's identity, grad mode) and whose body is
// one Op::Region: the compiled graph, called (from Python, which checks the
// containers among them) on the block's params and inputs, once what the
// trace read from elsewhere (see reads.rs) is seen to be as it was. A run whose guard fails falls through to the version
// the trace came from, which traces again (another variant) when it has run
// often enough again.
//
// A place (a block, or a span) has one region per state it was traced in: its
// variants (see Variants). The block has one region version, whose guard is
// that of the version the traces are made from; each variant has the checks
// its own trace added. Where the variants differ by something small that a
// call's args show at once (which object an arg or an item of a tuple arg
// is, the value of an int), they are found by that key instead of being
// asked in turn. A place holds up to max_regions() variants per key and up
// to max_keys() keys.
//
// A key whose object lives for one call (a cache object the caller makes on
// every call, pinned by identity) gives a variant that can never run again,
// and a place that is given such an object would trace one on every call and
// keep each call's object alive. So a place that, MAX_BARREN calls of the
// program in a row, traced under a key it didn't have and ran no variant,
// takes no further key, and drops the variants that never ran with all they
// hold (see Variants::room).
//
// A place may also be told apart by something that comes back, while what
// each variant is traced for does not (the image's number in a batch finds
// the key, the number of boxes found in that image is a size no later call
// has): its keys fill up with variants that never run again, each holding
// what its trace read. So a place where, of the variants that have had
// SETTLED calls of the program to run again, MIN_SETTLED or more exist and
// more than three in four never ran, takes no further variant at all and
// drops those that never ran.
//
// A variant keeps no object of the program alive that it is found or checked
// by (see pin.rs): one whose object died can never run again, and the place
// drops it when it is next entered (see Table::reap).
//
// A place belongs to a code object, and a code object can be the code of
// several functions (types.FunctionType(f.__code__, other_globals), a
// function made again from a code object that was kept), each with globals
// and builtins of its own. What a trace read of the frame's two namespaces is
// checked in the dicts of the frame it ran in (see reads.rs), so a variant is
// for the frames that have those two dicts and for no other (Variant::home):
// a frame with other namespaces is passed over by every variant traced
// elsewhere and gets its own trace.

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};

use pyo3::ffi;
use pyo3::ffi::PyObject;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use crate::cfg::ir::{Check, RegionOut, RegionSpec};
use crate::exec::ffi::{Frame, GilCell, tr_frame_builtins, tr_frame_globals};
use crate::exec::pin::Pin;
use crate::exec::types::Slot;

static ENABLED: AtomicBool = AtomicBool::new(false);
static OBSERVE: AtomicBool = AtomicBool::new(false);
static AFTER: AtomicU32 = AtomicU32::new(2);
static MAX_REGIONS: AtomicUsize = AtomicUsize::new(4);
static MAX_KEYS: AtomicUsize = AtomicUsize::new(64);
// Variants asked at a place's entry that their own checks of the args turned
// down (see _cfg_region_passed_over).
static PASSED_OVER: AtomicU64 = AtomicU64::new(0);
// How many times the program entered the executor from outside (see
// next_call): what "a call" is to a place.
static CALL: AtomicU64 = AtomicU64::new(0);
// Calls in a row in which a place traced under a new key and ran nothing,
// after which it takes no new key.
const MAX_BARREN: u32 = 3;
// Calls of the program after which a variant that has not run again counts
// against its place, and how many such variants a place is judged by.
const SETTLED: u64 = 32;
const MIN_SETTLED: usize = 16;

// A call of the program begins (call_with_cfg_execution).
#[inline]
pub fn next_call() {
    CALL.fetch_add(1, Ordering::Relaxed);
}
static CHECK_READS: AtomicBool = AtomicBool::new(true);

/// Whether a region is entered only while what its trace read is as it was
/// (see exec/reads.rs). On, except to measure what the check costs: a region
/// that isn't checked computes with what its trace read.
#[pyfunction]
pub fn _set_cfg_region_reads(on: bool) {
    CHECK_READS.store(on, Ordering::Relaxed);
}

/// Compile pure blocks into regions (see exec/region.rs) once a version has
/// run `after` times: up to `variants` for each key of a place (one per
/// input specialization) and up to `keys` keys (a loop body over different
/// modules has one per module); a place whose variants differ by nothing a
/// key is made of has one key.
#[pyfunction]
#[pyo3(signature = (on, after=2, variants=4, keys=64))]
pub fn _set_cfg_regions(on: bool, after: u32, variants: usize, keys: usize) {
    ENABLED.store(on, Ordering::Relaxed);
    AFTER.store(after.max(1), Ordering::Relaxed);
    MAX_REGIONS.store(variants.max(1), Ordering::Relaxed);
    MAX_KEYS.store(keys.max(1), Ordering::Relaxed);
}

/// How many times a variant of a place was asked at the place's entry and
/// turned down by its own checks of the args (before its reads were checked
/// or anything of it ran), over all places. A place whose variants are found
/// by their key adds nothing here; one whose variants differ only by tensor
/// metadata adds one for each variant in front of the one that runs.
#[pyfunction]
pub fn _cfg_region_passed_over() -> u64 {
    PASSED_OVER.load(Ordering::Relaxed)
}

/// Record the types of the args of the blocks and spans that would be
/// traced, for their traces (torch.vm's input_types), whether or not regions
/// are on.
#[pyfunction]
pub fn _set_cfg_region_observe(on: bool) {
    OBSERVE.store(on, Ordering::Relaxed);
}

#[inline]
pub fn observing() -> bool {
    OBSERVE.load(Ordering::Relaxed)
}

#[inline]
pub fn max_regions() -> usize {
    MAX_REGIONS.load(Ordering::Relaxed)
}

#[inline]
pub fn max_keys() -> usize {
    MAX_KEYS.load(Ordering::Relaxed)
}

#[inline]
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

// How many times a span's trace may give up before the span isn't traced any
// more. A run that leaves the span by a way no earlier run took says nothing
// about the runs that don't.
pub const MAX_GAVE_UP: u32 = 3;

// Whether block b's version (the last, the one traces are made from) should
// be traced on this run.
pub fn due(runs: &AtomicU32, off: &AtomicBool) -> bool {
    !off.load(Ordering::Relaxed) && runs.fetch_add(1, Ordering::Relaxed) + 1 == AFTER.load(Ordering::Relaxed)
}

// Joins a run's args (borrowed, NULL allowed) into their types.
pub unsafe fn observe(types: &mut Vec<Slot>, args: &[*mut PyObject]) {
    types.resize_with(args.len(), Slot::default);
    for (slot, a) in types.iter_mut().zip(args) {
        if !slot.skips(*a) {
            slot.observe(*a, false);
        }
    }
}

// Runs the body being traced on the given args (borrowed); Ok(its outputs,
// owned) or Err with the body's exception set.
pub type Body<'a> = dyn FnMut(&[*mut PyObject]) -> Result<Vec<*mut PyObject>, ()> + 'a;

thread_local! {
    static BODY: Cell<Option<*mut Body<'static>>> = const { Cell::new(None) };
    static TRACING: Cell<bool> = const { Cell::new(false) };
    // The exception by which the trace running on this thread was refused
    // while its body ran (see _region_refuse).
    static REFUSAL: std::cell::RefCell<Option<Py<PyAny>>> = const { std::cell::RefCell::new(None) };
}

/// The trace running on this thread is refused, by `why`, an exception the
/// caller raises into the traced code right after. That code may catch it
/// and go on: from here on the body is no run of the program's, so when it
/// ends without an error the executor raises `why` again, and the block
/// runs as it is.
#[pyfunction]
pub fn _region_refuse(why: Py<PyAny>) {
    REFUSAL.with_borrow_mut(|r| {
        r.get_or_insert(why);
    });
}

/// The trace running on this thread is given up and the executor runs its
/// code again: takes back what the body stored or appended (see writes.rs;
/// the executor's own call of writes::undo that follows finds nothing
/// left), and returns the addresses of the containers its instructions
/// filled in place (see covered.rs), for the tracer to give those among
/// its args back what they held (torch._rust_inductor._hand_back), which
/// must come after the writes are taken back.
#[pyfunction]
pub fn _region_given_up() -> Vec<usize> {
    super::writes::undo();
    super::covered::filled()
}

// Refuses the trace running on this thread from the executor, in place of
// what its body was about to do: the refusal (`why`: "it ...", as any) is
// the error raised there. `later`: the place is traced again (see
// covered.rs), which torch.vm.regions shows as "not yet".
pub fn refuse(py: Python<'_>, why: &str, later: bool) {
    match py.import("torch._rust_inductor").and_then(|m| m.call_method1("_refuse", (why, later))) {
        Ok(e) => PyErr::from_value(e).restore(py),
        Err(e) => e.restore(py),
    }
}

// Asked when a traced body is over, of what it hands on (`outs`, borrowed,
// NULL allowed): a stand-in for a float kept where `plain` cannot give the
// float back refuses the trace (torch._rust_inductor._kept_by says where),
// so that the run does not stand and the program never gets the stand-in.
// No exception is left set.
pub unsafe fn kept(outs: &[*mut PyObject]) {
    let py = Python::assume_attached();
    let handed: Vec<Bound<'_, PyAny>> = outs.iter().filter(|o| !o.is_null()).map(|o| Bound::from_borrowed_ptr(py, *o)).collect();
    let handed = PyList::new(py, handed);
    let by = super::outside_executor(|| {
        handed.and_then(|l| py.import("torch._rust_inductor")?.call_method1("_kept_by", (l,))).and_then(|k| k.extract::<Option<String>>())
    });
    if let Ok(Some(class)) = by {
        refuse(py, &format!("it hands a float argument on in a {class}, which a region cannot make again"), false);
        ffi::PyErr_Clear();
    }
}

// Whether the trace that runs (or the last that ran) on this thread was
// refused while its body ran.
pub fn refused() -> bool {
    REFUSAL.with_borrow(|r| r.is_some())
}

// Raises the refusal again, for a body that went on past it.
pub fn raise_refusal(py: Python<'_>) {
    if let Some(why) = REFUSAL.with_borrow(|r| r.as_ref().map(|w| w.clone_ref(py))) {
        PyErr::from_value(why.into_bound(py)).restore(py);
    }
}

// Whether a trace is running on this thread. Regions don't run then (the
// trace wouldn't see their ops), nor are others traced.
#[inline]
pub fn tracing() -> bool {
    TRACING.get()
}

// While a region is traced, the floats it computes with are instances of a
// float subclass (torch._rust_inductor._Scalar), so that the arithmetic on
// them is recorded. The traced run is also the call's own run, so what it
// leaves behind (a block's outputs; a span's locals, stack values and return
// value) is given to the program as floats: `o` (borrowed, NULL allowed) with
// each such instance replaced by the float it is, in `o` itself and in the
// exact tuples, lists and dicts it holds. A new reference, to `o` where there
// was nothing to replace.
pub unsafe fn plain(o: *mut PyObject) -> *mut PyObject {
    if o.is_null() {
        return o;
    }
    let py = Python::assume_attached();
    let made = py
        .import("torch._rust_inductor")
        .and_then(|m| m.getattr("_plain_result"))
        .and_then(|f| f.call1((Bound::from_borrowed_ptr(py, o),)));
    match made {
        Ok(x) => x.into_ptr(),
        Err(_) => {
            ffi::PyErr_Clear();
            ffi::Py_INCREF(o);
            o
        }
    }
}

fn null_sentinel(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    static NULL: pyo3::sync::PyOnceLock<Py<PyAny>> = pyo3::sync::PyOnceLock::new();
    NULL.get_or_try_init(py, || Ok::<_, PyErr>(py.import("torch._rust_inductor")?.getattr("NULL")?.unbind())).map(|n| n.bind(py).clone())
}

/// Runs the body of the block being traced (see torch._rust_inductor), once.
#[pyfunction]
#[pyo3(signature = (*args))]
pub fn _region_body<'py>(py: Python<'py>, args: &Bound<'py, PyTuple>) -> PyResult<Bound<'py, PyTuple>> {
    let Some(body) = BODY.take() else {
        return Err(pyo3::exceptions::PyRuntimeError::new_err("no region body is being traced"));
    };
    let null = null_sentinel(py)?;
    let objs: Vec<*mut PyObject> = args.iter().map(|a| if a.is(&null) { std::ptr::null_mut() } else { a.as_ptr() }).collect();
    // The trace runs outside the executor; the body runs in it again, so
    // that the frames it enters are the executor's and what they read is
    // recorded (see reads.rs).
    let outs = {
        let _exec = super::ExecGuard::new();
        let _recording = super::reads::Recording::start();
        let _own = super::ambient::Own::start();
        unsafe { (*body)(&objs) }
    };
    match outs {
        Ok(outs) => {
            let items: Vec<Bound<'py, PyAny>> = outs
                .into_iter()
                .map(|o| unsafe { Bound::from_owned_ptr_or_opt(py, o) }.unwrap_or_else(|| null.clone()))
                .collect();
            PyTuple::new(py, items)
        }
        Err(()) => Err(PyErr::fetch(py)),
    }
}

// A check a variant makes of its place's args (by position), beyond those of
// the guard its block's region version has.
pub enum Own {
    // The arg is this object.
    Is(u32, Pin),
    // On the arg, or on global state (None).
    Check(Option<u32>, Check),
}

// Where a fact is about: an arg, or (with an index other than WHOLE) an item
// of an arg that is an exact tuple or list.
pub type Pos = (u32, u32);
const WHOLE: u32 = u32::MAX;

// One region of a place: the state it was traced in (what its trace
// depended on among the place's args), and what runs then.
pub struct Variant {
    // A block's: what the trace added to the guard. A span's region makes
    // every check itself (see RegionSpec::quick and its func).
    pub checks: Vec<Own>,
    pub spec: RegionSpec,
    // What its checks (these, the quick ones, func's) pin to one object or
    // to one small value, by position (sorted): the atom the arg or item has
    // whenever they hold. A call whose atom there is another is not for this
    // variant.
    facts: Vec<(Pos, u64)>,
    // The objects of the facts, each for as long as the program keeps it:
    // an atom that is an address is compared only while they all live.
    pins: Vec<Pin>,
    // The globals and the builtins of the frame it was traced in (see
    // namespace): the dicts its reads of globals are checked in.
    home: [Pin; 2],
    // Whether it ever ran.
    ran: AtomicBool,
    // The call of the program it was traced in.
    born: u64,
}

impl Variant {
    // Whether a call with these args, in a frame with the namespaces `home`
    // (see namespaces), has the variant's facts: asked before anything else
    // of it, so that a variant for another object or value is passed over
    // whichever of its checks the fact comes from (func's, for a str or a
    // bound method), and one traced in a frame with other globals or
    // builtins whatever its checks say of the dicts it was traced in.
    pub unsafe fn facts_hold(&self, args: &[*mut PyObject], home: &[*mut PyObject; 2]) -> bool {
        self.home.iter().zip(home).all(|(mine, now)| mine.is(*now)) && !self.pins.iter().any(Pin::dead) && self.facts.iter().all(|(p, a)| atom_at(args, *p) == *a)
    }

    // Whether it can never run again: an object it is found by died (a
    // module whose dict was a namespace of its frame too), or one its reads
    // expect to find (see reads.rs).
    fn dead(&self) -> bool {
        self.pins.iter().chain(&self.home).any(Pin::dead) || self.spec.reads.as_ref().is_some_and(|r| r.get().dead())
    }
}

// The globals and the builtins of frame `f` (borrowed).
pub unsafe fn namespaces(f: *mut Frame) -> [*mut PyObject; 2] {
    [tr_frame_globals(f), tr_frame_builtins(f)]
}

// How a variant holds a namespace (the globals, the builtins) of the frame
// it was traced in. A dict takes no weak reference. One that is the dict of
// the module loaded under the name the dict gives itself
// (`sys.modules[d["__name__"]].__dict__ is d`: the globals of a function of
// a module, the builtins) is found through that module, held weakly (a
// module's dict is the same one for as long as the module lives), and the
// variant dies with the module. Any other (the globals of exec, of
// types.FunctionType, of a module that is no longer loaded) is counted for
// as long as the variant lives, with what it holds.
unsafe fn namespace(dict: *mut PyObject) -> Pin {
    let mut pin = Pin::counted(dict);
    let modules = ffi::PyImport_GetModuleDict();
    if ffi::PyDict_CheckExact(dict) != 0 && ffi::PyDict_CheckExact(modules) != 0 {
        let key = ffi::PyUnicode_InternFromString(c"__name__".as_ptr());
        if key.is_null() {
            ffi::PyErr_Clear();
            return pin;
        }
        let name = super::reads::dict_get(dict, key);
        ffi::Py_DECREF(key);
        let module = if !name.is_null() && ffi::PyUnicode_CheckExact(name) != 0 { super::reads::dict_get(modules, name) } else { std::ptr::null_mut() };
        if !module.is_null() && ffi::PyModule_Check(module) != 0 {
            // (Stays counted unless the module has this dict.)
            pin.through(&Pin::weak(module));
        }
    }
    pin
}

#[inline]
fn mix(h: u64, x: u64) -> u64 {
    (h.rotate_left(5) ^ x).wrapping_mul(0x517c_c1b7_2722_0a95)
}

// What a key is made of at one position: the value of an exact int or str
// (their hashes run no Python code); for a bound method what it is bound to
// (`layer.forward` is another method object on every read); else which
// object it is.
unsafe fn atom(x: *mut PyObject) -> u64 {
    if x.is_null() {
        1
    } else if ffi::PyLong_CheckExact(x) != 0 {
        mix(2, ffi::PyObject_Hash(x) as u64)
    } else if ffi::PyUnicode_CheckExact(x) != 0 {
        mix(3, ffi::PyObject_Hash(x) as u64)
    } else {
        let bound_to = super::ffi::tr_bound_self(x);
        if bound_to.is_null() { mix(4, x as usize as u64) } else { mix(5, bound_to as usize as u64) }
    }
}

unsafe fn atom_at(args: &[*mut PyObject], (i, j): Pos) -> u64 {
    let a = args.get(i as usize).copied().unwrap_or(std::ptr::null_mut());
    if j == WHOLE {
        return atom(a);
    }
    let j = j as ffi::Py_ssize_t;
    if a.is_null() {
        0
    } else if ffi::PyTuple_CheckExact(a) != 0 && j < ffi::PyTuple_GET_SIZE(a) {
        atom(ffi::PyTuple_GET_ITEM(a, j))
    } else if ffi::PyList_CheckExact(a) != 0 && j < ffi::PyList_GET_SIZE(a) {
        atom(ffi::PyList_GET_ITEM(a, j))
    } else {
        0
    }
}

#[derive(Default)]
struct Table {
    // In the order they were traced.
    all: Vec<Arc<Variant>>,
    // The key's positions: those every variant has a fact for.
    at: Vec<Pos>,
    // By key (the facts at `at`, hashed): indices into `all`, in order.
    // Empty while `at` is.
    by_key: HashMap<u64, Vec<u32>>,
    // The call (CALL) it was last entered in, and what happened in it: a
    // variant ran, a variant was traced under a key it didn't have.
    call: u64,
    ran: bool,
    new_key: bool,
    // Calls in a row, up to the last, with a new key and no run.
    barren: u32,
    // It takes no new key any more (see MAX_BARREN).
    closed: bool,
    // It takes no variant any more (see SETTLED).
    shut: bool,
}

impl Table {
    unsafe fn key_of(&self, args: &[*mut PyObject]) -> u64 {
        self.at.iter().fold(0, |h, p| mix(h, atom_at(args, *p)))
    }

    fn key(&self, v: &Variant) -> u64 {
        // Every position of `at` is one of v's facts.
        self.at.iter().fold(0, |h, p| mix(h, v.facts.iter().find(|f| f.0 == *p).map_or(0, |f| f.1)))
    }

    // Drops the variants that never ran. `settled`: because most of those
    // that had the time never ran again; otherwise because every call brought
    // a key the place didn't have.
    unsafe fn drop_unrun(&mut self, settled: bool) {
        let why = if settled { "settled" } else { "barren" };
        self.drop_where(|v| !v.ran.load(Ordering::Relaxed), |_| why);
    }

    // Drops the variants that can never run again because an object they
    // were traced for died. Asked when the place is entered: a pin per fact
    // of each variant, and what the reads of the variants asked since found.
    unsafe fn reap(&mut self) {
        if self.all.iter().any(|v| v.dead()) {
            // One that never ran, after MAX_BARREN calls in a row that each
            // brought a new key: the object is another on every call, and
            // the place is about to close (see room).
            let barren = self.barren >= MAX_BARREN;
            self.drop_where(Variant::dead, |v| if barren && !v.ran.load(Ordering::Relaxed) { "barren" } else { "gone" });
        }
    }

    unsafe fn drop_where(&mut self, gone: impl Fn(&Variant) -> bool, why: impl Fn(&Variant) -> &'static str) {
        let (dropped, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut self.all).into_iter().partition(|v| gone(v));
        self.all = kept;
        self.index();
        let py = Python::assume_attached();
        for v in dropped {
            // The record of its trace lets go of what the trace read. (The
            // engine's own code: no frame of the program's.)
            let told = super::outside_executor(|| py.import("torch._rust_inductor").and_then(|m| m.call_method1("_dropped", (v.spec.func.clone_ref(py), why(&v)))));
            if let Err(e) = told {
                e.write_unraisable(py, None);
            }
        }
    }

    // Entered: a new call of the program closes the account of the last.
    fn enter(&mut self) {
        let now = CALL.load(Ordering::Relaxed);
        if now != self.call {
            if self.ran {
                self.barren = 0;
            } else if self.new_key {
                self.barren += 1;
            }
            (self.call, self.ran, self.new_key) = (now, false, false);
        }
    }

    // The keys of the variants, anew: the positions every one of them has a
    // fact for, and each variant under its key.
    fn index(&mut self) {
        self.at = match self.all.first() {
            Some(first) => first.facts.iter().map(|f| f.0).filter(|p| self.all.iter().all(|v| v.facts.iter().any(|f| f.0 == *p))).collect(),
            None => vec![],
        };
        self.by_key.clear();
        if !self.at.is_empty() {
            for j in 0..self.all.len() {
                let k = self.key(&self.all[j]);
                self.by_key.entry(k).or_default().push(j as u32);
            }
        }
    }
}

// The variants of a place. Shared by every Cfg made from the one the place
// got its first region in; read and changed with the GIL held.
pub struct Variants(GilCell<Table>);

impl Default for Variants {
    fn default() -> Self {
        Variants(GilCell::new(Table::default()))
    }
}

impl Variants {
    pub fn len(&self) -> usize {
        unsafe { (*self.0.get()).all.len() }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn keys(&self) -> usize {
        unsafe { (*self.0.get()).by_key.len().max(1) }
    }

    // The i-th variant to ask for a call with these args (the newest first):
    // with a key, only those under the call's (computed once, into `key`).
    // One variant alone is asked without a key.
    pub unsafe fn candidate(&self, args: &[*mut PyObject], key: &mut Option<u64>, i: usize) -> Option<Arc<Variant>> {
        let t = &mut *self.0.get();
        if i == 0 {
            t.enter();
            t.reap();
        }
        let n = t.all.len();
        let j = if n <= 1 || t.at.is_empty() {
            n.checked_sub(i + 1)?
        } else {
            let k = *key.get_or_insert_with(|| t.key_of(args));
            let under = t.by_key.get(&k)?;
            under[under.len().checked_sub(i + 1)?] as usize
        };
        // Shared: it outlives the borrow of the table, which a trace made
        // under the call may add to and a closing place takes from.
        Some(t.all[j].clone())
    }

    // Variant v ran. In the call of the program it was traced in it runs
    // on what its trace ran on (the same objects, which may all be that
    // call's own: an iterator over that call's tensors): that says nothing
    // about a later call, and is no run to the place's account.
    pub fn ran(&self, v: &Variant) {
        if v.born == CALL.load(Ordering::Relaxed) {
            return;
        }
        v.ran.store(true, Ordering::Relaxed);
        unsafe { (*self.0.get()).ran = true };
    }

    // Whether the place may get another variant for a call with these args:
    // its key has fewer than max_regions(), and is one of at most max_keys().
    // A key the place doesn't have is turned away once the place has been
    // given new keys for MAX_BARREN calls in a row without any of its
    // variants running (the key is an object or a value that is another on
    // every call): the place is closed to new keys, and the variants that
    // never ran are dropped (torch._rust_inductor._dropped says so of each).
    pub unsafe fn room(&self, args: &[*mut PyObject]) -> bool {
        let t = &mut *self.0.get();
        t.enter();
        if t.shut {
            return false;
        }
        let now = CALL.load(Ordering::Relaxed);
        let settled = t.all.iter().filter(|v| v.born + SETTLED <= now);
        let (n, never) = settled.fold((0, 0), |(n, never), v| (n + 1, never + usize::from(!v.ran.load(Ordering::Relaxed))));
        if n >= MIN_SETTLED && never * 4 > n * 3 {
            (t.shut, t.closed) = (true, true);
            t.drop_unrun(true);
            return false;
        }
        // A place whose variants each died with the object they were traced
        // for (see reap) has none left to tell a key by: its calls each
        // brought a new key all the same.
        if t.all.is_empty() && t.barren >= MAX_BARREN {
            t.closed = true;
        }
        if t.at.is_empty() {
            // (A closed place whose variants were all dropped has no key
            // positions left, and stays as it is.)
            return !t.closed && t.all.len() < max_regions();
        }
        if let Some(under) = t.by_key.get(&t.key_of(args)) {
            return under.len() < max_regions();
        }
        if t.closed {
            return false;
        }
        if t.barren < MAX_BARREN {
            return t.by_key.len() < max_keys();
        }
        t.closed = true;
        t.drop_unrun(false);
        false
    }

    pub fn push(&self, v: Variant) {
        let t = unsafe { &mut *self.0.get() };
        let at: Vec<Pos> = if t.all.is_empty() { v.facts.iter().map(|f| f.0).collect() } else { t.at.iter().copied().filter(|p| v.facts.iter().any(|f| f.0 == *p)).collect() };
        t.all.push(Arc::new(v));
        if at != t.at {
            // A variant without one of the positions: the keys are of those
            // left.
            t.at = at;
            t.by_key.clear();
        }
        if t.at.is_empty() {
            return;
        }
        let from = if t.by_key.is_empty() { 0 } else { t.all.len() - 1 };
        for j in from..t.all.len() {
            let k = t.key(&t.all[j]);
            let under = t.by_key.entry(k).or_default();
            under.push(j as u32);
            // The newest is the first under its key: a key the place didn't
            // have.
            if j + 1 == t.all.len() && under.len() == 1 {
                t.new_key = true;
            }
        }
    }
}

impl std::fmt::Debug for Variants {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} variants, {} keys", self.len(), self.keys())
    }
}

// Counts a variant asked at entry whose own checks turned the call down.
#[inline]
pub fn passed_over() {
    PASSED_OVER.fetch_add(1, Ordering::Relaxed);
}

// What a trace produced.
pub struct Traced {
    pub variant: Variant,
}

fn parse_own(t: &Bound<'_, PyTuple>, n_args: usize) -> PyResult<Own> {
    let bad = || pyo3::exceptions::PyValueError::new_err(format!("bad check {t}"));
    Ok(match super::parse_check_tuple(t)? {
        super::Parsed::Is(i, o) if i < n_args => Own::Is(i as u32, unsafe { Pin::weak(o.as_ptr()) }),
        super::Parsed::Check(None, c) => Own::Check(None, c),
        super::Parsed::Check(Some(i), c) if i < n_args => Own::Check(Some(i as u32), c),
        _ => return Err(bad()),
    })
}

// (arg, item or None, the object or value the checks pin it to): see
// torch._rust_inductor._keys.
unsafe fn parse_facts(keys: &[Bound<'_, PyTuple>]) -> PyResult<(Vec<(Pos, u64)>, Vec<Pin>)> {
    let mut facts: Vec<(Pos, u64)> = vec![];
    let mut held = vec![];
    for k in keys {
        let i: u32 = k.get_item(0)?.extract()?;
        let j: Option<u32> = k.get_item(1)?.extract()?;
        let want = k.get_item(2)?;
        let pos = (i, j.unwrap_or(WHOLE));
        if facts.iter().all(|f| f.0 != pos) {
            facts.push((pos, atom(want.as_ptr())));
            // What the atom is the address of: a bound method is another
            // object on every read, what it is bound to is not.
            let bound_to = super::ffi::tr_bound_self(want.as_ptr());
            held.push(Pin::weak(if bound_to.is_null() { want.as_ptr() } else { bound_to }));
        }
    }
    facts.sort_by_key(|f| f.0);
    Ok((facts, held))
}

// Traces a run of a block's version. `args`: its params then inputs
// (borrowed, NULL allowed), of which those at `unread` aren't read by the
// version's instrs (NULL if not used at all), and `types` their types over
// the runs observed (see observing); `body` runs it in frame `f`. Ok(None):
// not worth a region. With `span` (a span's run, which no guard checks
// first), the region makes all its checks itself; with `asked` (a span a cfg
// pass asked for) it is a region however few ops it runs.
#[allow(clippy::too_many_arguments)]
pub unsafe fn trace(name: &str, args: &[*mut PyObject], types: &[Slot], unread: &[u32], span: bool, asked: bool, f: *mut Frame, body: &mut Body<'_>) -> PyResult<Option<Traced>> {
    let py = Python::assume_attached();
    let null = null_sentinel(py)?;
    let items: Vec<Bound<'_, PyAny>> =
        args.iter().map(|a| Bound::from_borrowed_ptr_or_opt(py, *a).unwrap_or_else(|| null.clone())).collect();
    let n_args = args.len();
    let args = PyTuple::new(py, items)?;
    let types = (0..args.len())
        .map(|i| match types.get(i).and_then(|s| s.ty.as_ref()) {
            Some(t) => t.to_py(py),
            None => Ok(py.None().into_bound(py)),
        })
        .collect::<PyResult<Vec<_>>>()?;
    let body: *mut Body<'_> = body;
    BODY.set(Some(std::mem::transmute::<*mut Body<'_>, *mut Body<'static>>(body)));
    TRACING.set(true);
    REFUSAL.set(None);
    super::reads::begin();
    super::backward::ready();
    super::writes::begin();
    super::hist::begin();
    let r = py.import("torch._rust_inductor").and_then(|m| m.call_method1("trace_region", (name, args, unread.to_vec(), span, types, asked)));
    super::reads::end();
    TRACING.set(false);
    BODY.set(None);
    // A refusal is asked about until the next trace begins (see refused):
    // what it keeps then is its text, not the frames it was raised through
    // and the tensors in them.
    REFUSAL.with_borrow(|e| {
        if let Some(e) = e {
            ffi::PyException_SetTraceback(e.as_ptr(), ffi::Py_None());
            ffi::PyException_SetContext(e.as_ptr(), std::ptr::null_mut());
        }
    });
    let r = r?;
    if r.is_none() {
        return Ok(None);
    }
    #[allow(clippy::type_complexity)]
    let (func, outputs, objects, checks, reads, quick, keys, writes): (
        Bound<'_, PyAny>,
        Vec<Bound<'_, PyAny>>,
        Vec<Bound<'_, PyAny>>,
        Vec<Bound<'_, PyTuple>>,
        Option<Py<super::reads::Reads>>,
        Vec<Bound<'_, PyTuple>>,
        Vec<Bound<'_, PyTuple>>,
        Vec<Bound<'_, PyAny>>,
    ) = r.extract()?;
    let writes = writes.iter().map(parse_out).collect::<PyResult<Vec<_>>>()?;
    let quick = quick.iter().map(parse_quick).collect::<PyResult<Vec<_>>>()?;
    let outputs = outputs.iter().map(parse_out).collect::<PyResult<Vec<_>>>()?;
    // A span's region makes the checks itself (func, and the quick ones).
    let checks = if span { vec![] } else { checks.iter().map(|c| parse_own(c, n_args)).collect::<PyResult<Vec<_>>>()? };
    let (facts, held) = parse_facts(&keys)?;
    Ok(Some(Traced {
        variant: Variant {
            checks,
            spec: RegionSpec {
                func: func.unbind(),
                outputs,
                objects: objects.into_iter().map(Bound::unbind).collect(),
                writes,
                name: name.to_string(),
                reads,
                quick,
            },
            facts,
            pins: held,
            home: namespaces(f).map(|dict| namespace(dict)),
            ran: AtomicBool::new(false),
            born: CALL.load(Ordering::Relaxed),
        },
    }))
}

// A check of a region's args that its func makes too, asked first (see
// torch._rust_inductor._quick).
pub enum Quick {
    // Arg i is this object (true), or the same value by exact::same.
    Arg(u32, bool, Pin),
    // Arg i is an exact tuple or list whose item j is this (likewise).
    Item(u32, u32, bool, Pin),
    // These args, as a tuple, pass the TreeChecks.
    Tensors(Vec<u32>, Py<super::TreeChecks>),
    // Arg i (a container of tensors and plain values) passes the TreeChecks.
    Tree(u32, Py<super::TreeChecks>),
}

fn parse_quick(q: &Bound<'_, PyTuple>) -> PyResult<Quick> {
    let kind: String = q.get_item(0)?.extract()?;
    Ok(match kind.as_str() {
        "arg" => Quick::Arg(q.get_item(1)?.extract()?, q.get_item(2)?.extract()?, unsafe { Pin::weak(q.get_item(3)?.as_ptr()) }),
        "item" => Quick::Item(q.get_item(1)?.extract()?, q.get_item(2)?.extract()?, q.get_item(3)?.extract()?, unsafe { Pin::weak(q.get_item(4)?.as_ptr()) }),
        "tensors" => Quick::Tensors(q.get_item(1)?.extract()?, q.get_item(2)?.extract()?),
        "tree" => Quick::Tree(q.get_item(1)?.extract()?, q.get_item(2)?.extract()?),
        _ => return Err(pyo3::exceptions::PyValueError::new_err(format!("bad quick check {q}"))),
    })
}

impl Quick {
    // No Python code runs: a value is compared by exact::same (a Tree's
    // TreeChecks is one that is `quick`).
    unsafe fn holds(&self, py: Python<'_>, args: &[*mut PyObject]) -> bool {
        // A value compared by exact::same is of a class that takes no weak
        // reference (an int, a str, None): its pin never dies.
        let is = |x: *mut PyObject, same: bool, want: &Pin| if same { want.is(x) } else { want.live().is_some_and(|w| super::exact::same(x, w)) };
        let arg = |i: u32| args.get(i as usize).copied().unwrap_or(std::ptr::null_mut());
        match self {
            Quick::Arg(i, same, want) => is(arg(*i), *same, want),
            Quick::Item(i, j, same, want) => {
                let (a, j) = (arg(*i), *j as ffi::Py_ssize_t);
                if a.is_null() {
                    false
                } else if ffi::PyTuple_CheckExact(a) != 0 && j < ffi::PyTuple_GET_SIZE(a) {
                    is(ffi::PyTuple_GET_ITEM(a, j), *same, want)
                } else if ffi::PyList_CheckExact(a) != 0 && j < ffi::PyList_GET_SIZE(a) {
                    is(ffi::PyList_GET_ITEM(a, j), *same, want)
                } else {
                    false
                }
            }
            Quick::Tensors(idx, checks) => {
                if idx.iter().any(|i| arg(*i).is_null()) {
                    return false;
                }
                let Ok(t) = PyTuple::new(py, idx.iter().map(|i| Bound::from_borrowed_ptr(py, arg(*i)))) else {
                    ffi::PyErr_Clear();
                    return false;
                };
                checks.get().passes(t.as_any())
            }
            Quick::Tree(i, checks) => {
                let a = arg(*i);
                !a.is_null() && checks.get().passes(&Bound::from_borrowed_ptr(py, a))
            }
        }
    }
}

fn parse_out(o: &Bound<'_, PyAny>) -> PyResult<RegionOut> {
    let t = o.cast::<PyTuple>()?;
    let kind: String = t.get_item(0)?.extract()?;
    Ok(match kind.as_str() {
        "arg" => RegionOut::Arg(t.get_item(1)?.extract()?),
        "out" => RegionOut::Out(t.get_item(1)?.extract()?),
        "obj" => RegionOut::Obj(t.get_item(1)?.extract()?),
        "null" => RegionOut::Null,
        "tuple" | "list" => {
            let items = t.get_item(1)?.try_iter()?.map(|x| parse_out(&x?)).collect::<PyResult<Vec<_>>>()?;
            if kind == "tuple" { RegionOut::Tuple(items) } else { RegionOut::List(items) }
        }
        "dict" => {
            let keys: Vec<u32> = t.get_item(1)?.extract()?;
            let items = t.get_item(2)?.try_iter()?.map(|x| parse_out(&x?)).collect::<PyResult<Vec<_>>>()?;
            RegionOut::Dict(keys, items)
        }
        "iter" => RegionOut::Iter(Box::new(parse_out(&t.get_item(1)?)?)),
        "call" => {
            let items = t.get_item(2)?.try_iter()?.map(|x| parse_out(&x?)).collect::<PyResult<Vec<_>>>()?;
            RegionOut::Call(t.get_item(1)?.extract()?, items)
        }
        _ => return Err(pyo3::exceptions::PyValueError::new_err(format!("bad region output {o}"))),
    })
}

// Runs an Op::Region on `args` (borrowed: the block's params, then the
// version's inputs): Some(the block's outputs, owned), None if the region's
// own checks failed (run the version it was traced from), or Err with an
// exception set.
pub unsafe fn call(spec: &RegionSpec, args: &[*mut PyObject]) -> Result<Option<Vec<*mut PyObject>>, ()> {
    let py = Python::assume_attached();
    // Under a mode, a transform or a tracer of the program the code runs
    // as written: its calls are to be seen one by one.
    if let Some(why) = super::ambient::not_neutral() {
        if std::env::var_os("TORCH_RUST_REGION_DEBUG").is_some() {
            eprintln!("region {}: not entered: {why}", spec.name);
        }
        return Ok(None);
    }
    // The region of another state of its place (another input size) is
    // passed over here, for next to nothing.
    if !spec.quick.iter().all(|q| q.holds(py, args)) {
        passed_over();
        return Ok(None);
    }
    // Before anything of the region runs.
    if let Some(reads) = &spec.reads
        && CHECK_READS.load(Ordering::Relaxed)
        && !reads.get().check()
    {
        if std::env::var_os("TORCH_RUST_REGION_DEBUG").is_some() {
            eprintln!("region {}: miss on {}", spec.name, reads.get().last());
        }
        return Ok(None);
    }
    // A lowering bound to the region's constants is bound anew first if the
    // data of one moved (see reads.rs).
    if let Some(reads) = &spec.reads
        && reads.get().rebind(py).is_err()
    {
        return Err(());
    }
    let Ok(null) = null_sentinel(py) else { return Err(()) };
    let list = ffi::PyList_New(args.len() as ffi::Py_ssize_t);
    if list.is_null() {
        return Err(());
    }
    for (i, a) in args.iter().enumerate() {
        let a = if a.is_null() { null.as_ptr() } else { *a };
        ffi::Py_INCREF(a);
        ffi::PyList_SET_ITEM(list, i as ffi::Py_ssize_t, a);
    }
    // The compiled graph's Python wrappers run on CPython.
    let r = super::outside_executor(|| ffi::PyObject_CallOneArg(spec.func.as_ptr(), list));
    ffi::Py_DECREF(list);
    if r.is_null() {
        return Err(());
    }
    if r == ffi::Py_None() {
        ffi::Py_DECREF(r);
        return Ok(None);
    }
    let outs = if ffi::PyList_CheckExact(r) != 0 || ffi::PyTuple_CheckExact(r) != 0 { r } else { ffi::PySequence_Fast(r, c"region outputs".as_ptr()) };
    if outs.is_null() {
        ffi::Py_DECREF(r);
        return Err(());
    }
    let item = |k: u32| -> *mut PyObject {
        if ffi::PyList_CheckExact(outs) != 0 { ffi::PyList_GET_ITEM(outs, k as ffi::Py_ssize_t) } else { ffi::PyTuple_GET_ITEM(outs, k as ffi::Py_ssize_t) }
    };
    let n = ffi::PySequence_Size(outs);
    unsafe fn make(spec: &RegionSpec, o: &RegionOut, args: &[*mut PyObject], item: &dyn Fn(u32) -> *mut PyObject, n: isize) -> *mut PyObject {
        match o {
            RegionOut::Arg(i) => {
                let a = args[*i as usize];
                if !a.is_null() {
                    ffi::Py_INCREF(a);
                }
                a
            }
            RegionOut::Out(k) => {
                if *k as isize >= n {
                    ffi::PyErr_SetString(ffi::PyExc_RuntimeError, c"CFG executor: a region returned too few outputs".as_ptr());
                    return std::ptr::null_mut();
                }
                let x = item(*k);
                ffi::Py_INCREF(x);
                x
            }
            RegionOut::Obj(i) => {
                let x = spec.objects[*i as usize].as_ptr();
                ffi::Py_INCREF(x);
                x
            }
            RegionOut::Null => std::ptr::null_mut(),
            RegionOut::Call(f, items) => {
                let list = ffi::PyList_New(items.len() as isize);
                for (i, it) in items.iter().enumerate() {
                    let x = make(spec, it, args, item, n);
                    if x.is_null() {
                        ffi::Py_DECREF(list);
                        return std::ptr::null_mut();
                    }
                    ffi::PyList_SET_ITEM(list, i as isize, x);
                }
                let r = ffi::PyObject_CallOneArg(spec.objects[*f as usize].as_ptr(), list);
                ffi::Py_DECREF(list);
                r
            }
            RegionOut::Tuple(items) | RegionOut::List(items) => {
                let tuple = matches!(o, RegionOut::Tuple(_));
                let c = if tuple { ffi::PyTuple_New(items.len() as isize) } else { ffi::PyList_New(items.len() as isize) };
                for (i, it) in items.iter().enumerate() {
                    let x = make(spec, it, args, item, n);
                    if x.is_null() {
                        ffi::Py_DECREF(c);
                        return std::ptr::null_mut();
                    }
                    if tuple {
                        ffi::PyTuple_SET_ITEM(c, i as isize, x);
                    } else {
                        ffi::PyList_SET_ITEM(c, i as isize, x);
                    }
                }
                c
            }
            RegionOut::Dict(keys, items) => {
                let d = ffi::PyDict_New();
                for (k, it) in keys.iter().zip(items) {
                    let x = make(spec, it, args, item, n);
                    if x.is_null() || ffi::PyDict_SetItem(d, spec.objects[*k as usize].as_ptr(), x) < 0 {
                        ffi::Py_XDECREF(x);
                        ffi::Py_DECREF(d);
                        return std::ptr::null_mut();
                    }
                    ffi::Py_DECREF(x);
                }
                d
            }
            RegionOut::Iter(items) => {
                let c = make(spec, items, args, item, n);
                if c.is_null() {
                    return c;
                }
                let it = ffi::PyObject_GetIter(c);
                ffi::Py_DECREF(c);
                it
            }
        }
    }
    let mut result = Vec::with_capacity(spec.outputs.len());
    let mut ok = true;
    for o in &spec.outputs {
        let x = make(spec, o, args, &item, n);
        if x.is_null() && !matches!(o, RegionOut::Null | RegionOut::Arg(_)) {
            ok = false;
            break;
        }
        result.push(x);
    }
    // The graph ran and every output is made: now the run's writes, in the
    // order the trace made them (see writes.rs).
    ok = ok && super::writes::replay(&spec.writes, &spec.objects, &|o| make(spec, o, args, &item, n));
    if outs != r {
        ffi::Py_DECREF(outs);
    }
    ffi::Py_DECREF(r);
    if !ok {
        result.into_iter().for_each(|x| ffi::Py_XDECREF(x));
        return Err(());
    }
    Ok(Some(result))
}
