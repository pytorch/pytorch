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
use crate::ir::*;
use crate::pyffi::{self, new_ref, py_bool, xdecref, Obj, PyObj, Shared};
use crate::{frontend, inline, unroll};
use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use crate::spec::{self, PassCx, Pipeline, Unit, ValueObs};
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
    /// Unroll loops over a small constant range.
    pub unroll: bool,
    /// Compile the Python functions that compiled code calls.
    pub callees: bool,
    /// Take the registered shortcuts through library code (see shortcuts.rs).
    pub shortcuts: bool,
    /// Time each callee both compiled and interpreted, and leave it to the
    /// interpreter if that is where it runs faster.
    pub demote: bool,
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
            callees: true,
            shortcuts: true,
            demote: true,
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
    pub respec_left: u32,
    pub profile_runs: u32,
    pub next_version_id: u32,
    pub execs: u64,
    pub spec_execs: u64,
    pub entry_misses: u64,
    pub deopts: u64,
    pub rejected: Option<String>,
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
    trial: std::cell::Cell<Trial>,
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
    calls: u32,
    compiled_best: u64,
    compiled_samples: u8,
    settled_best: u64,
    settled_samples: u8,
    interpreted_best: u64,
    interpreted_samples: u8,
}

impl Trial {
    fn start(epoch: u32) -> Trial {
        Trial {
            phase: TRIAL_WARMUP,
            epoch,
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
/// A callee whose measurement never gets quiet enough stays compiled.
const TRIAL_MAX_CALLS: u32 = 400;
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

pub fn plan_changed() {
    PLAN_EPOCH.fetch_add(1, Ordering::Relaxed);
}

static TRIAL_DRAW: AtomicU64 = AtomicU64::new(0x9E37_79B9_7F4A_7C15);

unsafe impl Send for SettledBlocks {}
unsafe impl Sync for SettledBlocks {}

/// Per-name caches; only touched with the GIL held.
struct LoadCaches(Vec<std::cell::Cell<LoadCache>>);

unsafe impl Send for LoadCaches {}
unsafe impl Sync for LoadCaches {}

#[inline(always)]
unsafe fn dict_version(dict: Obj) -> u64 {
    (*(dict as *mut ffi::PyDictObject)).ma_version_tag
}

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
        function.unrolled = unroll::unroll_loops(&mut function);
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
/// torch.cfg_compile, or from anything compiled under it, each compiled the
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
    call_name: PyObj,
}

unsafe impl Send for Callees {}
unsafe impl Sync for Callees {}

const MAX_CALLEES: usize = 4096;
const RECENT_CALLEES: usize = 256;
/// Stands for "runs in the interpreter" in the recent-callee slots.
const INTERPRETED: *const Compiled = 1 as *const Compiled;

impl Callees {
    pub fn new(table: OpcodeTable) -> Callees {
        Callees {
            table,
            entries: Shared::new(HashMap::new()),
            recent: (0..RECENT_CALLEES).map(|_| std::cell::Cell::new((0, 0, std::ptr::null()))).collect(),
            interpreted: Shared::new(HashMap::new()),
            call_name: unsafe {
                PyObj::from_owned(ffi::PyUnicode_InternFromString(c"__call__".as_ptr()))
            },
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
    }
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
/// instruction pointer on the bytecode of the instruction it executes; the
/// locals of the frame stay empty, since they live in the VM's registers.
#[derive(Clone, Copy)]
struct Frame {
    /// Null where frames are not supported.
    ptr: *mut std::ffi::c_void,
    instr: *mut *mut u16,
    code: *mut u16,
}

impl Frame {
    const NONE: Frame = Frame { ptr: null_mut(), instr: null_mut(), code: null_mut() };

    unsafe fn push(function: Obj) -> Result<Frame, ()> {
        let mut frame = Frame::NONE;
        if let Some(push) = capi::api().and_then(|api| api.frame_push) {
            frame.ptr = push(function, &mut frame.instr, &mut frame.code);
            if frame.ptr.is_null() {
                return Err(());
            }
        }
        Ok(frame)
    }

    /// `kept` (consumed, may be null) goes on living with the frame object,
    /// if there is one that outlives the frame.
    unsafe fn pop(self, kept: Obj) {
        if self.ptr.is_null() {
            xdecref(kept);
        } else {
            (capi::api().unwrap_unchecked().frame_pop.unwrap_unchecked())(self.ptr, kept);
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
            self.frames[self.depth].pop(null_mut());
        }
        self.depth -= 1;
    }
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
        let py = pyo3::Python::assume_gil_acquired();
        match py.eval_bound("lambda: None", None, None) {
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
    let result = ffi::PyObject_CallNoArgs(nothing);
    if result.is_null() {
        return false;
    }
    ffi::Py_DECREF(result);
    true
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
    /// Kernel results that were kept unboxed, and how many of those got a
    /// Python object after all.
    pub unboxed_results: AtomicU64,
    pub boxed_later: AtomicU64,
    /// Function name and file name of each origin, for traceback entries.
    origin_c: Vec<(CString, CString)>,
    /// Per handler block: the registers that survive entering it, which are
    /// what it reads and the frame slots.
    handler_keep: Vec<Option<analysis::BitSet>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Generic,
    Profile,
    Spec,
}

const MAX_CAPTURE_ATTEMPTS: u32 = 8;

type OpResult = Result<Obj, ()>;

unsafe fn set_error(exc: Obj, msg: String) {
    let c = CString::new(msg).unwrap_or_default();
    ffi::PyErr_SetString(exc, c.as_ptr());
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
            panic!("cfg_compile: could not create the Python object of a tensor");
        }
        *regs.get_unchecked_mut(v as usize) = boxed;
        self.boxed_later.fetch_add(1, Ordering::Relaxed);
        boxed
    }

    pub fn new(
        func: Function,
        options: Options,
        pipeline: Arc<Pipeline>,
        callees: Arc<Callees>,
    ) -> Compiled {
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
                respec_left: options.max_respecializations,
                profile_runs: 0,
                next_version_id: 1,
                execs: 0,
                spec_execs: 0,
                entry_misses: 0,
                deopts: 0,
                rejected: None,
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
                calls: std::cell::Cell::new(0),
                trial: std::cell::Cell::new(Trial::start(PLAN_EPOCH.load(Ordering::Relaxed))),
            },
            generic,
            state: Shared::new(state),
            options,
            pipeline,
            frames: Shared::new(Vec::new()),
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
                    let result = self.exec(&mut regs, frame);
                    // The traceback of an exception that leaves the function
                    // refers to its frame, and through it the interpreter
                    // keeps the locals alive: so do the registers here.
                    let mut kept = null_mut();
                    if result.is_null() && frame.held() {
                        // The values that were bound to a variable, not the
                        // temporaries: the exception itself is one of those,
                        // and would be kept by its own traceback.
                        let live: Vec<Obj> = regs[..nregs - 1]
                            .iter()
                            .zip(&self.func.value_names)
                            .filter(|(o, name)| {
                                name.is_some() && !o.is_null() && !capi::is_unboxed(**o)
                            })
                            .map(|(o, _)| *o)
                            .collect();
                        kept = ffi::PyTuple_New(live.len() as isize);
                        for (i, o) in live.iter().enumerate() {
                            if !kept.is_null() {
                                ffi::PyTuple_SET_ITEM(kept, i as isize, new_ref(*o));
                            }
                        }
                    }
                    // Before the registers are released, as the interpreter
                    // unlinks a frame before it clears its locals.
                    frame.pop(kept);
                    result
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
            frames.leave();
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
        if !self.options.callees {
            return None;
        }
        let (func, receiver) = if ffi::PyFunction_Check(callee) != 0 {
            (callee, null_mut())
        } else if ffi::Py_TYPE(callee) == std::ptr::addr_of_mut!(pyffi::PyMethod_Type) {
            (pyffi::PyMethod_Function(callee), pyffi::PyMethod_Self(callee))
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
            (if target.is_null() { call } else { target }, callee)
        };
        if func.is_null() || ffi::PyFunction_Check(func) == 0 {
            return None;
        }
        self.compiled_function(func).map(|compiled| (compiled, func, receiver))
    }

    unsafe fn compiled_function(&self, func: Obj) -> Option<*const Compiled> {
        let callees = &self.callees;
        let code = ffi::PyFunction_GetCode(func);
        let globals = ffi::PyFunction_GetGlobals(func);
        let slot = callees.recent.get_unchecked((code as usize >> 4) & (RECENT_CALLEES - 1));
        let (recent_code, recent_globals, recent) = slot.get();
        if recent_code == code as usize && recent_globals == globals as usize {
            return if recent == INTERPRETED { None } else { Some(recent) };
        }
        let found = self
            .compiled_function_slow(func, code, globals)
            .filter(|compiled| (**compiled).settled.trial.get().phase != TRIAL_DEMOTED);
        slot.set((code as usize, globals as usize, found.unwrap_or(INTERPRETED)));
        found
    }

    unsafe fn compiled_function_slow(&self, func: Obj, code: Obj, globals: Obj) -> Option<*const Compiled> {
        let callees = &self.callees;
        let key = (code as usize, globals as usize);
        if let Some(c) = callees.entries.borrow().get(&key) {
            return Some(Arc::as_ptr(&c.compiled));
        }
        if callees.interpreted.borrow().contains_key(&(code as usize))
            || callees.entries.borrow().len() >= MAX_CALLEES
        {
            return None;
        }
        let py = pyo3::Python::assume_gil_acquired();
        let bound = pyo3::Bound::from_borrowed_ptr(py, func);
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
                Some(pointer)
            }
            // Unsupported by the frontend: the interpreter runs it.
            Err(_) => {
                callees.interpreted.borrow_mut().insert(code as usize, PyObj::from_borrowed(code));
                None
            }
        }
    }

    /// `call` for the function that was handed to torch.cfg_compile, which
    /// is measured against the interpreter as its callees are.
    pub unsafe fn call_measured(&self, function: Obj, args: Obj, kwargs: Obj) -> Obj {
        let phase = self.settled.trial.get().phase;
        if phase == TRIAL_COMPILED || !self.options.demote {
            return self.call(function, args, kwargs);
        }
        if phase == TRIAL_DEMOTED {
            return ffi::PyObject_Call(function, args, kwargs);
        }
        self.trial_call(
            self,
            null_mut(),
            || self.call(function, args, kwargs),
            || ffi::PyObject_Call(function, args, kwargs),
        )
    }

    /// Whether it is still open which of the two this function is run as.
    pub fn is_measuring(&self) -> bool {
        let phase = self.settled.trial.get().phase;
        self.options.demote && (phase == TRIAL_WARMUP || phase == TRIAL_SAMPLING)
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
        let slot = &(*compiled).settled.trial;
        let mut trial = slot.get();
        if trial.phase == TRIAL_DEMOTED {
            return run_interpreted();
        }
        let epoch = PLAN_EPOCH.load(Ordering::Relaxed);
        if trial.epoch != epoch {
            trial = Trial::start(epoch);
        }
        trial.calls += 1;
        slot.set(trial);
        let events = TRIAL_EVENTS.fetch_add(1, Ordering::Relaxed) + 1;
        if trial.phase == TRIAL_WARMUP {
            if trial.calls >= TRIAL_WARMUP_CALLS {
                slot.set(Trial { phase: TRIAL_SAMPLING, ..trial });
            }
            return run_compiled();
        }
        // Which of the two a call measures is drawn, not alternated: calls
        // often come in a pattern (two layers of a model, one after the
        // other), and alternating would give each side one half of it.
        let as_compiled = if trial.interpreted_samples >= TRIAL_SAMPLES {
            true
        } else if trial.settled_samples >= TRIAL_SAMPLES {
            false
        } else {
            let mut state = TRIAL_DRAW.load(Ordering::Relaxed);
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            TRIAL_DRAW.store(state, Ordering::Relaxed);
            state & 1 == 0
        };
        let start = std::time::Instant::now();
        let result = if as_compiled { run_compiled() } else { run_interpreted() };
        let elapsed = start.elapsed().as_nanos() as u64;
        // The callee may have called itself: what is in the slot now counts.
        let mut trial = slot.get();
        if trial.phase != TRIAL_SAMPLING {
            return result;
        }
        if !result.is_null() {
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
        let quarter = |t: u64| t + (t >> 2);
        let margin = trial.interpreted_best >> TRIAL_MARGIN_SHIFT;
        let seen = |samples: u8, enough: u8| samples >= enough && trial.interpreted_samples >= enough;
        let won = (seen(trial.compiled_samples, TRIAL_EARLY_SAMPLES)
            && quarter(trial.compiled_best) < trial.interpreted_best)
            || (seen(trial.compiled_samples, TRIAL_SAMPLES)
                && trial.interpreted_best + margin >= trial.compiled_best);
        let lost = (seen(trial.settled_samples, TRIAL_EARLY_SAMPLES)
            && quarter(trial.interpreted_best) < trial.settled_best)
            || (seen(trial.settled_samples, TRIAL_SAMPLES)
                && trial.interpreted_best + margin < trial.settled_best);
        if won {
            trial.phase = TRIAL_COMPILED;
        } else if lost {
            trial.phase = TRIAL_DEMOTED;
            if !func.is_null() {
                // Calls stop finding the compiled form.
                let code = ffi::PyFunction_GetCode(func);
                let globals = ffi::PyFunction_GetGlobals(func);
                let recent =
                    self.callees.recent.get_unchecked((code as usize >> 4) & (RECENT_CALLEES - 1));
                recent.set((code as usize, globals as usize, INTERPRETED));
            }
        } else if trial.calls >= TRIAL_MAX_CALLS {
            trial.phase = TRIAL_COMPILED;
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
                    (dict_version(f.globals.as_ptr()), dict_version(f.builtins.as_ptr()));
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
            // The block loop runs these two; it has the frames.
            Op::Hold(_) | Op::FrameEnter(_) | Op::FrameLeave => Ok(null_mut()),
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
                    let missed =
                        if name == "__exit__" { " (missed __exit__ method)" } else { "" };
                    set_error(
                        ffi::PyExc_TypeError,
                        format!(
                            "'{}' object does not support the context manager protocol{}",
                            pyffi::type_name(o),
                            missed
                        ),
                    );
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
            Op::MakeFunction { code, defaults, kwdefaults, annotations, closure } => {
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
                        let version = dict_version(dict);
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
                let converted = match conv {
                    1 => ffi::PyObject_Str(val),
                    2 => ffi::PyObject_Repr(val),
                    3 => ffi::PyObject_ASCII(val),
                    _ => new_ref(val),
                };
                if converted.is_null() {
                    return Err(());
                }
                if *spec == NONE && ffi::PyUnicode_CheckExact(converted) != 0 {
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
                    set_error(
                        ffi::PyExc_ValueError,
                        if size > n {
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
                Ok(py_bool(
                    o == f.consts.get_unchecked(*func as usize).as_ptr()
                        && ffi::PyFunction_GetCode(o)
                            == f.consts.get_unchecked(*code as usize).as_ptr(),
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
                            "cfg_compile: tensor_size ran without its guard".to_string(),
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
                            "cfg_compile: tensor_dim ran without its guard".to_string(),
                        );
                        Err(())
                    }
                }
            }
            Op::Dispatch(d) => match capi::api() {
                Some(api) => self.exec_dispatch(regs, &d.slot, &d.operands, &d.fallback, api),
                None => self.exec_op(regs, &d.fallback),
            },
            Op::Guard(_) => Ok(null_mut()),
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
    ) -> (OpResult, *mut capi::TcfgSite, Option<String>) {
        let Some(operands) = self.site_operands(kind, op) else {
            return (self.exec_op(regs, op), null_mut(), None);
        };
        // Operands are borrowed from the registers, which the operation does
        // not touch; read them first so a call sees its callee.
        let ptrs_before = self.operand_ptrs(regs, kind, op, &operands);
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
        let mut site = null_mut();
        let mut reason = None;
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
            (api.trace_free)(trace);
        }
        xdecref(method);
        (result, site, reason)
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
        if n < self.options.max_site_variants && misses < MAX_CAPTURE_ATTEMPTS as u64 * 4 {
            let (result, site, _) = self.capture_exec(regs, fallback, slot.kind, api);
            if !site.is_null() {
                slot.push(site);
            }
            return result;
        }
        self.exec_op(regs, fallback)
    }

    /// One instruction in the profiling tier.
    unsafe fn exec_profiled(
        &self,
        b: usize,
        index: usize,
        inst: &Inst,
        regs: &mut [Obj],
        api: &TcfgApi,
    ) -> OpResult {
        let (slot, attempts) = {
            let st = self.state.borrow();
            let bs = &st.blocks[b];
            (bs.slots[index].clone(), bs.capture_attempts[index])
        };
        let result = if let Some(slot) = slot {
            let ptrs = self
                .site_operands(slot.kind, &inst.op)
                .and_then(|operands| self.operand_ptrs(regs, slot.kind, &inst.op, &operands));
            match ptrs {
                Some(ptrs) => self.exec_slot(regs, &slot, ptrs.as_ptr(), 0, &inst.op, api),
                None => self.exec_op(regs, &inst.op),
            }
        } else if attempts >= MAX_CAPTURE_ATTEMPTS {
            self.exec_op(regs, &inst.op)
        } else {
            match self.site_kind(regs, &inst.op, api) {
                None => self.exec_op(regs, &inst.op),
                Some(kind) => {
                    let (result, site, reason) = self.capture_exec(regs, &inst.op, kind, api);
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
        if let Ok(obj) = result {
            if inst.dst != NONE {
                self.state.borrow_mut().obs[inst.dst as usize].observe(obj);
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
        // One version that is entered unconditionally, or none and no more
        // attempts at one: from now on there is nothing to decide.
        if bs.versions.len() == 1 && bs.versions[0].entry_guards.is_empty() {
            settle(Arc::as_ptr(&bs.versions[0]), true);
        } else if bs.versions.is_empty() && bs.respec_left == 0 {
            settle(generic, false);
        }
        for version in &bs.versions {
            if version.entry_guards.iter().all(|g| Self::guard_holds(g, regs, api)) {
                bs.spec_execs += 1;
                return (Arc::as_ptr(version), Mode::Spec);
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
        let (slots, obs, speculate, id) = {
            let mut st = self.state.borrow_mut();
            let obs = st.obs.clone();
            let bs = &mut st.blocks[b];
            if bs.respec_left == 0 {
                return;
            }
            bs.respec_left -= 1;
            bs.profile_runs += 1;
            let id = bs.next_version_id;
            bs.next_version_id += 1;
            // The last attempt must hold for every input.
            (bs.slots.clone(), obs, bs.respec_left > 0, id)
        };
        let mut unit = Unit::from_generic(&self.func, b, speculate);
        let cx = PassCx { cf: self, slots: &slots, obs: &obs };
        let outcome = match self.pipeline.run(&mut unit, &cx) {
            Ok(true) => spec::finalize(unit, self, id).map(Some),
            Ok(false) => Ok(None),
            Err(err) => {
                // A failing pass must not take the program down with it: the
                // block simply stays generic.
                let msg = pyo3::Python::with_gil(|py| err.value_bound(py).to_string());
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

    // ------------------------------------------------------------------
    // The interpreter loop
    // ------------------------------------------------------------------

    unsafe fn exec(&self, regs: &mut [Obj], frame: Frame) -> Obj {
        let mut frames = Frames::new(frame);
        let result = self.exec_blocks(regs, &mut frames);
        // Every path out of an inlined function pops its frame; this is for
        // the one that was not thought of.
        while frames.depth > 0 {
            frames.leave();
        }
        result
    }

    unsafe fn exec_blocks(&self, regs: &mut [Obj], frames: &mut Frames) -> Obj {
        let api = capi::api();
        let func = &self.func;
        if self.options.specialize {
            self.settled.calls.set(self.settled.calls.get() + 1);
        }
        let mut cur: usize = 0;
        let mut pred: usize = usize::MAX;
        let mut back_edges: u32 = 0;
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
                    if back_edges & 63 == 0 && !tick() {
                        self.fail(frames, block.bc_start);
                        raised!('blocks, cur);
                    }
                }
            }

            let (mut code, mut mode) = self.select(cur, regs, api);
            let mut pc: usize = 0;
            let next: usize = loop {
                let mut resume: Option<usize> = None;
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
                            Op::FrameEnter(function) => {
                                let function = func.consts.get_unchecked(*function as usize).as_ptr();
                                frames.enter(function).map(|()| null_mut())
                            }
                            Op::FrameLeave => {
                                frames.leave();
                                Ok(null_mut())
                            }
                            Op::CaughtLasti => Ok(frames.top().lasti()),
                            op => {
                                frames.top().at(inst.bc & OFFSET_MASK);
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
                                raised!('blocks, cur);
                            }
                        }
                        pc += 1;
                    }
                    if resume.is_none() {
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
