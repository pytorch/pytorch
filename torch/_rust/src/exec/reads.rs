// What a region's trace read, and the check that it still reads the same.
//
// A region is the tensor ops of one traced run. Whatever else that run read
// from something that outlives it is baked in: a scalar as a constant of the
// graph, a branch as the path taken, a callee as the code that ran. The
// block's guard covers the block's own inputs; this covers the rest, at any
// call depth: while the body is traced (see region.rs), the executor runs
// every frame under it and reports here
//   - a global load: the dict that had the name (or, for a builtin, also that
//     the globals don't have it), the name, the value;
//   - an attribute or method load (and getattr, hasattr): the object's class
//     by identity, what the class (with its bases) has under that name and
//     under __getattribute__ and __getattr__ (so that a method, a property
//     or a class attribute bound anew is seen), and the instance's __dict__
//     (or the module's) by item, present or absent; a slot or another C data
//     descriptor is read again by the check;
//   - a call, a subscript, an iteration, a length or a truth test of an
//     instance of a Python class: the object's class, and what the class has
//     as the special method that ran;
//   - a subscript or a membership test of a dict (str or int key) or a list
//     (int index), and dict.get: the item (of a defaultdict that stored the
//     key because it was asked for it: the item it stored);
//   - a cell read: the cell's content;
//   - a tensor found by one of these reads that the run then asks something
//     of (an attribute, a method, its length, its class) or hands to C code,
//     itself or inside a tuple, a list, a set or a dict: the tensor's key
//     (exec/tensor_key.rs) as it was when it was first asked (see Keyed);
//   - a frame of a Python function: which code the function has, and the
//     defaults a call fills its arguments from (the tuple by identity, the
//     keyword-only ones also item by item);
//   - an attribute found in (or missing from) an object's __dict__: which
//     dict the object has;
//   - a call of a functools.partial: its function and arguments, and its
//     keywords whole;
//   - a question about a class (isinstance, issubclass, type(obj), and what
//     a metaclass's own methods end in): the object's class, the order that
//     class resolves its bases in, the metaclass of the class asked about,
//     and for a class with ABCMeta's machinery abc's counter of
//     registrations (see asked.rs);
//   - an object of a Python class handed to a builtin or to an instruction
//     the interpreter implements in C (float(obj), `obj == 1`, f"{obj}"):
//     the object's class, and what the class has under each name C code
//     resolves (see special.rs);
//   - any other instruction or builtin handed a dict, a list, a set or a dict
//     view (iterating it, unpacking it, slicing it, zip, sorted, ...): the
//     whole content, in order (an OrderedDict's own order, which is not the
//     order of the dict's table); len() and truth: the length. C code reads
//     through what it is handed (the list inside the tuple of a subscript,
//     the rows of a nested list made into a tensor): the lists, dicts, sets
//     and tuples inside a container or a tuple are read whole too, and what
//     those hold.
// A tuple or a frozenset itself isn't recorded (its items can't be bound
// anew); the containers it holds are.
//
// Values are compared by exact type and value if they are an int, a float
// (by bits) or a str (exec/exact.rs), and by identity otherwise. The entries are grouped by
// holder; a dict or a class whose CPython version tag is the one it had when
// it last passed is skipped whole, so a check that passes costs one
// comparison per dict and per class. A class is compared by what its names
// give, not by the tag alone: fx binds nn.Module's __call__ and __getattr__
// to its own while a region is traced and puts them back, which changes the
// tag of every module class and nothing a region depends on.
//
// The thread state a graph is specialized on and no instruction reads
// (autocast, the default dtype, deterministic algorithms) is one integer,
// taken when the trace starts and compared first.
//
// Two kinds of entries are dropped rather than checked. One whose holder
// nothing else refers to: nothing can change it. And one whose holder is a
// list, dict or set the traced run made (made.rs), whoever still refers to
// it: a region's run makes its own. They keep the traced call's temporaries
// and results from being kept alive, and from failing every check.
//
// While the trace runs the entries count a reference to each object (an
// entry is found again by the object's address). Once it is over they hold
// by pin.rs: weakly where the object's class takes a weak reference, and an
// instance's __dict__ through the instance, so that a region keeps nothing
// alive that the program dropped. An entry whose holder then died is
// nothing to check and is dropped (no later call can come by the holder,
// as above); an object an entry expects to FIND that died can never be
// found again, and the region is dead (see Reads::dead). What is still
// counted: see the list at `weaken`. An entry
// of anything else is never dropped: one that is not as it was read when the
// traced body is over refuses the trace, unless a write the run recorded
// explains it (see _region_settled), because what changed it is something
// no region would do again; and one that changes later is a later call's
// miss.
//
// The writes of the traced run to what outlives it are recorded by writes.rs:
// a region makes them again, in order, on every run. Here is what they mean
// for the reads. What the run reads of an item after it wrote that item is
// what it wrote, and no entry. What it read of a container (or an object)
// before writing to it stays as it was read, so the region is entered only
// in the state the trace found before its writes; a run that changes what
// it read (a counter) therefore finds a state no later call has again, and
// every call gets a region of its own until the place closes. A container
// read whole after the run changed it is refused.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};

use pyo3::ffi;
use pyo3::ffi::PyObject;
use pyo3::prelude::*;

use super::asked;
use super::exact;
use super::ffi::*;
use super::overlap::Overlap;
use super::pin::{self, Pin};
use super::special;
use crate::bindings::ffi as tffi;
use super::tensor_key::TensorKey;

// What a tensor answers about itself that a region neither checks nor
// computes, so that a trace that asks is refused: what autograd keeps on it,
// which memory and which object it is, how often it was written, and the
// marks a region does not look at (conjugate, negative, zero, names,
// coalesced). A region finds a tensor argument by its class, dtype, device,
// sizes, strides and requires_grad alone, and makes the tensors of its block
// anew. The one list of these: exec/shown.rs takes the methods among them
// out of torch's functions that only compute.
pub(super) const UNCHECKED_OF_A_TENSOR: &[&str] = &[
    "grad",
    "grad_fn",
    "_grad",
    "_grad_fn",
    "_base",
    "is_leaf",
    "output_nr",
    "retains_grad",
    "_version",
    "_cdata",
    "data_ptr",
    "storage_offset",
    "untyped_storage",
    "storage",
    "is_set_to",
    "is_shared",
    "is_pinned",
    "is_inference",
    "_is_view",
    "is_conj",
    "is_neg",
    "_is_zerotensor",
    "has_names",
    "is_coalesced",
    "__dict__",
];

// Entries a region may have, and items a container read whole may have.
const MAX_ENTRIES: usize = 1 << 15;
const MAX_ITEMS: usize = 1 << 12;
// A round of the sweep drops what only the records hold; what was held by
// something a round dropped goes in the next (a call's arguments, held by the
// cells of a closure, held by the closure).
const SWEEP_ROUNDS: usize = 64;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Want {
    Absent,
    Present,
    // The same object.
    Is,
    // An int, float or str of the same type and value.
    Equal,
}

struct Item {
    key: Py<PyAny>,
    want: Want,
    value: Option<Pin>,
}

struct Whole {
    // A dict's keys then values pairwise; a list's or a set's items.
    items: Vec<Pin>,
    // Where it was read (for describe).
    at: String,
}

#[derive(Clone, Copy, PartialEq)]
enum Holder {
    Dict,
    List,
    Set,
}

struct Group {
    holder: Pin,
    kind: Holder,
    // The dict's version tag when the group last held.
    version: Cell<u64>,
    items: Vec<Item>,
    whole: Option<Whole>,
    len: Option<isize>,
    // Read whole (or measured) after the traced run changed it.
    late: bool,
    late_at: String,
    // A namespace a global was loaded from: there before any run.
    root: bool,
}

struct Typed {
    obj: Pin,
    class: Pin,
}

// A class and the tuple it resolves its bases in.
struct Mro {
    class: Pin,
    mro: Py<PyAny>,
}

struct Class {
    class: Pin,
    // The class's version tag when it last held.
    tag: Cell<u32>,
    // What the class has under each name looked up (None: nothing), found
    // when the trace is over.
    names: Vec<(Py<PyAny>, Option<Pin>)>,
}

struct Read {
    // A cell (name None) or an object and the attribute a C descriptor gives.
    obj: Pin,
    name: Option<Py<PyAny>>,
    want: Want,
    value: Option<Pin>,
    // Where an attribute was read, for the refusal if it can't be checked.
    at: String,
}

// A Python function a frame of which ran under the traced body: the code
// that ran, and the defaults its call was filled from, are the function's
// for as long as it has them (`f.__code__ = ...`, `f.__defaults__ = ...`).
struct Function {
    func: Pin,
    code: Pin,
    defaults: Option<Pin>,
    kwdefaults: Option<Pin>,
}

// An object and the __dict__ its attributes were read from: the items read
// are the object's for as long as it has that dict (`obj.__dict__ = {...}`).
struct Dicted {
    obj: Pin,
    dict: Pin,
}

// A tensor the traced run found by a read (an item, a cell, an attribute:
// state, not an argument) and asked something of: its dtype or shape, whether
// it requires grad, its length, its class, a method. The answer is in the
// region as a constant or as the path taken, and the block's guard covers the
// region's arguments only: the region is entered while the tensor has the
// key it had when it was first asked. It lives as long as an entry that
// leads to the tensor (see led): the read that found it says it is still
// the one.
struct Keyed {
    obj: Pin,
    key: TensorKey,
    // See Asked.
    ours: bool,
}

// A tensor the run asked something of, while the trace runs: most are the
// run's own (an argument, a result), so it is held weakly, and a tensor made
// later at the address of one that died is asked anew.
struct Asked {
    weak: Py<PyAny>,
    // None: it has no key (see exec/tensor_key.rs).
    key: Option<TensorKey>,
    // An input of the trace or the result of an operator it recorded (see
    // _region_settled): the guard's to compare, or the graph's to make.
    ours: bool,
}

#[derive(Default)]
struct Entries {
    groups: Vec<Group>,
    typed: Vec<Typed>,
    classes: Vec<Class>,
    reads: Vec<Read>,
    functions: Vec<Function>,
    dicted: Vec<Dicted>,
    mros: Vec<Mro>,
    // abc's counter of registrations when a class with ABCMeta's machinery
    // was asked about.
    abc: Option<u64>,
    keyed: Vec<Keyed>,
}

impl Entries {
    fn len(&self) -> usize {
        let groups: usize = self.groups.iter().map(|g| g.items.len() + g.whole.as_ref().map_or(0, |w| w.items.len()) + usize::from(g.len.is_some())).sum();
        let asked = self.mros.len() + usize::from(self.abc.is_some());
        groups + self.typed.len() + self.classes.iter().map(|c| c.names.len()).sum::<usize>() + self.reads.len() + self.functions.len() + self.dicted.len() + asked + self.keyed.len()
    }
}

// The reads of a trace in progress.
#[derive(Default)]
struct Trace {
    thread: u64,
    entries: Entries,
    group_of: HashMap<usize, usize>,
    class_of: HashMap<usize, usize>,
    seen: HashSet<(usize, usize)>,
    refused: Option<String>,
    count: usize,
    // The items (holder, key) the traced run wrote (see writes.rs), and the
    // containers it changed.
    written: HashSet<(usize, usize)>,
    changed: HashSet<usize>,
    // The containers the trace keeps its own records in (which tensor is
    // which proxy): what is read from them or written to them while the body
    // runs (the entry of a tensor that died is taken out) is not the body's.
    own: HashSet<usize>,
    // The defaultdicts a load of the traced run stored a missing key into,
    // with where.
    filled: HashMap<usize, String>,
    // The tensors the run asked something of, in order, and each one's place
    // by address.
    asked_of: Vec<Asked>,
    asked_at: HashMap<usize, usize>,
    // What was no longer as it was read when the body was over, by the
    // address of what it was read from (see _region_settled).
    stale: Vec<(Stale, usize, String)>,
    // The tuples and containers one handing to C code has walked (see
    // through).
    walked: HashSet<usize>,
}

// What a stale read was read from: a cell, a function (its code and
// defaults), an object (its __dict__), a container (an item, all of it, its
// length).
enum Stale {
    Cell,
    Function,
    Dict,
    Group,
}

thread_local! {
    static TRACE: RefCell<Option<Trace>> = const { RefCell::new(None) };
    // Whether the frames running now are the traced body's.
    static ON: Cell<bool> = const { Cell::new(false) };
    // Frames the eval-frame hook has seen on this thread.
    static FRAMES: Cell<u64> = const { Cell::new(0) };
    // Whether an instr of the traced body (or of a frame under it) is
    // running: a frame entered then is one the body called. One entered
    // between instrs (a finalizer or a weakref callback, run when the
    // executor drops a value) isn't the body's, and isn't recorded.
    static IN_INSTR: Cell<bool> = const { Cell::new(false) };
}

// Whether a frame entered now is to record what it reads.
#[inline]
pub fn entered_by_body() -> bool {
    ON.get() && IN_INSTR.get()
}

#[inline]
pub fn in_instr() -> bool {
    IN_INSTR.get()
}

#[inline]
pub fn set_in_instr(on: bool) -> bool {
    IN_INSTR.replace(on)
}

#[inline]
pub fn on() -> bool {
    ON.get()
}

#[inline]
pub fn frames() -> u64 {
    FRAMES.get()
}

#[inline]
pub fn frame_entered() {
    FRAMES.set(FRAMES.get().wrapping_add(1));
}

// Starts the reads of a trace on this thread.
pub fn begin() {
    // Looked up here, outside the executor: not a read of the body's.
    builtins();
    super::made::clear();
    asked::init();
    super::built::init();
    TRACE.set(Some(Trace { thread: tffi::region_thread_state(), ..Trace::default() }));
}

pub fn end() {
    super::made::clear();
    TRACE.set(None);
    ON.set(false);
}

// Records while it lives: around the traced body, and around what the
// trace's own handlers call back into.
pub struct Recording(bool);

impl Recording {
    pub fn start() -> Recording {
        let active = TRACE.with_borrow(|t| t.is_some());
        Recording(ON.replace(active))
    }

    pub fn pause() -> Recording {
        Recording(ON.replace(false))
    }
}

impl Drop for Recording {
    fn drop(&mut self) {
        ON.set(self.0);
    }
}

fn with(f: impl FnOnce(&mut Trace)) {
    TRACE.with(|t| match t.try_borrow_mut() {
        Ok(mut t) => {
            if let Some(t) = t.as_mut().filter(|t| t.refused.is_none()) {
                f(t);
                if t.count > MAX_ENTRIES {
                    t.refused = Some(format!("it depends on more than {MAX_ENTRIES} reads"));
                }
                if let Some(why) = t.refused.as_mut() {
                    why.push_str(&unsafe { place() });
                }
            }
        }
        // A read made while one is being recorded isn't the body's: the
        // collector ran a finalizer.
        Err(_) => {}
    });
}

// " (in QUALNAME, line N)" for the frame running now.
unsafe fn place() -> String {
    let f = tr_current_frame(ffi::PyThreadState_Get());
    if f.is_null() {
        return String::new();
    }
    let line = PyCode_Addr2Line(tr_frame_code(f), tr_frame_get_instr(f) * 2);
    format!(" (in {}, line {line})", text(tr_frame_qualname(f)))
}

// The trace can't be a region: it depends on something no check covers.
pub fn refuse(why: String) {
    with(|t| t.refused = Some(why));
}

unsafe fn exact_scalar(o: *mut PyObject) -> bool {
    let tp = ffi::Py_TYPE(o);
    tp == &raw mut ffi::PyLong_Type || tp == &raw mut ffi::PyFloat_Type || tp == &raw mut ffi::PyUnicode_Type
}

unsafe fn key_ok(k: *mut PyObject) -> bool {
    let tp = ffi::Py_TYPE(k);
    tp == &raw mut ffi::PyLong_Type || tp == &raw mut ffi::PyUnicode_Type
}

unsafe fn want_of(o: *mut PyObject) -> Want {
    if exact_scalar(o) { Want::Equal } else { Want::Is }
}

pub(super) unsafe fn own(o: *mut PyObject) -> Py<PyAny> {
    Py::from_borrowed_ptr(Python::assume_attached(), o)
}

// As an entry holds an object while the trace runs.
unsafe fn counted(o: *mut PyObject) -> Pin {
    Pin::counted(o)
}

// Whether `now` (borrowed, NULL: nothing there) is what was read. What is
// wanted Equal is an exact int, float or str (want_of): the one value it is
// to every reader (exec/exact.rs).
unsafe fn same(now: *mut PyObject, want: Want, value: Option<&Pin>) -> bool {
    match want {
        Want::Absent => now.is_null(),
        Want::Present => !now.is_null(),
        Want::Is => value.is_some_and(|v| v.is(now)),
        // (Its pin stays counted. Nothing there is not what was read.)
        Want::Equal => !now.is_null() && value.and_then(Pin::live).is_some_and(|v| exact::same(now, v)),
    }
}

// What a tuple or a frozenset holds (borrowed), or None for anything else:
// the two builtin containers whose items cannot be bound anew, so that what
// leads to one leads to what it holds, and no entry is of one.
unsafe fn fixed(o: *mut PyObject) -> Option<Vec<*mut PyObject>> {
    if ffi::PyTuple_Check(o) != 0 {
        Some((0..ffi::PyTuple_GET_SIZE(o)).map(|i| ffi::PyTuple_GET_ITEM(o, i)).collect())
    } else if ffi::PyAnySet_Check(o) != 0 && ffi::PySet_Check(o) == 0 {
        Some(snapshot(o, Holder::Set))
    } else {
        None
    }
}

pub(super) unsafe fn tensor(o: *mut PyObject) -> bool {
    !tffi::tensor_by_type(o as usize).is_null()
}

pub(super) unsafe fn text(o: *mut PyObject) -> String {
    let mut n: ffi::Py_ssize_t = 0;
    let p = if ffi::PyUnicode_Check(o) != 0 { ffi::PyUnicode_AsUTF8AndSize(o, &mut n) } else { std::ptr::null() };
    if p.is_null() {
        ffi::PyErr_Clear();
        return "?".into();
    }
    String::from_utf8_lossy(std::slice::from_raw_parts(p as *const u8, n as usize)).into_owned()
}

pub(super) unsafe fn type_name(o: *mut PyObject) -> String {
    std::ffi::CStr::from_ptr((*ffi::Py_TYPE(o)).tp_name).to_string_lossy().into_owned()
}

// For a refusal: an int, a float or a str as it prints (C code), anything
// else by its class; NULL is nothing.
unsafe fn shown(o: *mut PyObject) -> String {
    if o.is_null() {
        return "nothing".into();
    }
    if !exact_scalar(o) {
        return format!("a {}", type_name(o));
    }
    let r = ffi::PyObject_Repr(o);
    if r.is_null() {
        ffi::PyErr_Clear();
        return "?".into();
    }
    let out = text(r);
    ffi::Py_DECREF(r);
    out
}

// The item of dict d at key k (borrowed), NULL if absent; no Python code runs
// for a str or int key.
pub(super) unsafe fn dict_get(d: *mut PyObject, k: *mut PyObject) -> *mut PyObject {
    let x = ffi::PyDict_GetItemWithError(d, k);
    if x.is_null() && !ffi::PyErr_Occurred().is_null() {
        ffi::PyErr_Clear();
    }
    x
}

// Whether every key of dict d is a str or an int: finding one again (as an
// OrderedDict does for each step of its iterator) runs no Python code.
unsafe fn plain_keys(d: *mut PyObject) -> bool {
    let (mut pos, mut k, mut v) = (0, std::ptr::null_mut(), std::ptr::null_mut());
    while ffi::PyDict_Next(d, &mut pos, &mut k, &mut v) != 0 {
        if !key_ok(k) {
            return false;
        }
    }
    true
}

// Whether what `holder` holds can be compared in the order C code reads it
// in: an OrderedDict only while its keys are plain (see snapshot).
unsafe fn ordered_ok(holder: *mut PyObject, kind: Holder) -> bool {
    kind != Holder::Dict || tr_ordered(holder) == 0 || plain_keys(holder)
}

// What `holder` holds, in the order its own C code hands it out. An
// OrderedDict's order is not its table's (move_to_end changes the one and
// not the other): its keys are taken from the iterator of OrderedDict
// itself. Call only when ordered_ok.
unsafe fn snapshot(holder: *mut PyObject, kind: Holder) -> Vec<*mut PyObject> {
    let mut out = vec![];
    match kind {
        Holder::Dict if tr_ordered(holder) != 0 => {
            let keys = tr_ordered_keys(holder);
            while !keys.is_null() {
                let k = ffi::PyIter_Next(keys);
                if k.is_null() {
                    break;
                }
                // The dict holds the key as well.
                ffi::Py_DECREF(k);
                let v = dict_get(holder, k);
                if v.is_null() {
                    break;
                }
                out.extend([k, v]);
            }
            ffi::Py_XDECREF(keys);
            ffi::PyErr_Clear();
        }
        Holder::Dict => {
            let (mut pos, mut k, mut v) = (0, std::ptr::null_mut(), std::ptr::null_mut());
            while ffi::PyDict_Next(holder, &mut pos, &mut k, &mut v) != 0 {
                out.push(k);
                out.push(v);
            }
        }
        Holder::List => out.extend((0..ffi::PyList_GET_SIZE(holder)).map(|i| ffi::PyList_GET_ITEM(holder, i))),
        Holder::Set => {
            let (mut pos, mut k, mut h) = (0, std::ptr::null_mut(), 0);
            while _PySet_NextEntry(holder, &mut pos, &mut k, &mut h) != 0 {
                out.push(k);
            }
        }
    }
    out
}

unsafe fn size(holder: *mut PyObject, kind: Holder) -> isize {
    match kind {
        Holder::Dict => ffi::PyDict_Size(holder),
        Holder::List => ffi::PyList_GET_SIZE(holder),
        Holder::Set => ffi::PySet_Size(holder),
    }
}

impl Item {
    unsafe fn holds(&self, holder: *mut PyObject, kind: Holder) -> bool {
        let now = match kind {
            Holder::Dict => dict_get(holder, self.key.as_ptr()),
            Holder::List => {
                let i = ffi::PyLong_AsSsize_t(self.key.as_ptr());
                if i >= 0 && i < ffi::PyList_GET_SIZE(holder) { ffi::PyList_GET_ITEM(holder, i) } else { std::ptr::null_mut() }
            }
            Holder::Set => std::ptr::null_mut(),
        };
        same(now, self.want, self.value.as_ref())
    }
}

impl Whole {
    unsafe fn holds(&self, holder: *mut PyObject, kind: Holder) -> bool {
        if !ordered_ok(holder, kind) {
            return false;
        }
        let now = snapshot(holder, kind);
        now.len() == self.items.len() && now.iter().zip(&self.items).all(|(a, b)| b.live().is_some_and(|b| *a == b || (exact_scalar(*a) && exact::same(*a, b))))
    }
}

impl Group {
    unsafe fn holds(&self) -> bool {
        let h = self.holder.as_ptr();
        self.len.is_none_or(|n| size(h, self.kind) == n) && self.items.iter().all(|i| i.holds(h, self.kind)) && self.whole.as_ref().is_none_or(|w| w.holds(h, self.kind))
    }

    fn empty(&self) -> bool {
        self.items.is_empty() && self.whole.is_none() && self.len.is_none() && !self.late
    }
}

impl Class {
    // Whether the class still has what it had under each name.
    unsafe fn holds(&self) -> bool {
        let tp = self.class.as_ptr() as *mut ffi::PyTypeObject;
        self.names.iter().all(|(name, had)| pin::was(had, tr_type_lookup(tp, name.as_ptr())))
    }
}

impl Mro {
    // The same classes in the same order: bases bound back make an equal
    // tuple, not the same one.
    unsafe fn holds(&self) -> bool {
        let (now, was) = ((*(self.class.as_ptr() as *mut ffi::PyTypeObject)).tp_mro, self.mro.as_ptr());
        let n = ffi::PyTuple_GET_SIZE(was);
        now == was || (!now.is_null() && ffi::PyTuple_GET_SIZE(now) == n && (0..n).all(|i| ffi::PyTuple_GET_ITEM(now, i) == ffi::PyTuple_GET_ITEM(was, i)))
    }
}

impl Function {
    // Three fields of the function object: no attribute is looked up.
    unsafe fn holds(&self) -> bool {
        let f = self.func.as_ptr() as *mut ffi::PyFunctionObject;
        self.code.is((*f).func_code) && pin::was(&self.defaults, (*f).func_defaults) && pin::was(&self.kwdefaults, (*f).func_kwdefaults)
    }
}

impl Read {
    unsafe fn holds(&self) -> bool {
        match &self.name {
            None => same(PyCell_Get_borrowed(self.obj.as_ptr()), self.want, self.value.as_ref()),
            Some(name) => {
                // The data descriptor's own answer: no __getattribute__ of
                // the class runs.
                let now = ffi::PyObject_GenericGetAttr(self.obj.as_ptr(), name.as_ptr());
                if now.is_null() {
                    ffi::PyErr_Clear();
                }
                let ok = same(now, self.want, self.value.as_ref());
                ffi::Py_XDECREF(now);
                ok
            }
        }
    }
}

// A cell's content, borrowed (the cell keeps it alive while the GIL is held).
#[allow(non_snake_case)]
unsafe fn PyCell_Get_borrowed(cell: *mut PyObject) -> *mut PyObject {
    let x = PyCell_Get(cell);
    ffi::Py_XDECREF(x);
    x
}

impl Trace {
    unsafe fn group(&mut self, holder: *mut PyObject, kind: Holder) -> &mut Group {
        let i = match self.group_of.get(&(holder as usize)) {
            Some(i) => *i,
            None => {
                self.entries.groups.push(Group { holder: counted(holder), kind, version: Cell::new(0), items: vec![], whole: None, len: None, late: false, late_at: String::new(), root: false });
                self.group_of.insert(holder as usize, self.entries.groups.len() - 1);
                self.entries.groups.len() - 1
            }
        };
        &mut self.entries.groups[i]
    }

    // The first read of an item is the one recorded: a later one that differs
    // is of something the traced run changed.
    unsafe fn item(&mut self, holder: *mut PyObject, kind: Holder, key: *mut PyObject, want: Want, value: *mut PyObject) {
        // What the run wrote itself is what it reads back.
        if self.written.contains(&(holder as usize, key as usize)) || self.own.contains(&(holder as usize)) {
            return;
        }
        let value = (!value.is_null() && matches!(want, Want::Is | Want::Equal)).then(|| counted(value));
        if !self.seen.insert((holder as usize, key as usize)) {
            // `key in d` and then `d[key]`: the value is what is checked.
            if value.is_some()
                && let Some(i) = self.group(holder, kind).items.iter_mut().find(|i| i.key.as_ptr() == key && i.want == Want::Present)
            {
                i.want = want;
                i.value = value;
            }
            return;
        }
        self.count += 1;
        self.group(holder, kind).items.push(Item { key: own(key), want, value });
    }

    unsafe fn dict_item(&mut self, d: *mut PyObject, key: *mut PyObject, present_only: bool) {
        let now = dict_get(d, key);
        let want = if now.is_null() {
            Want::Absent
        } else if present_only {
            Want::Present
        } else {
            want_of(now)
        };
        self.item(d, Holder::Dict, key, want, now);
    }

    unsafe fn whole(&mut self, holder: *mut PyObject, kind: Holder) {
        if self.group_of.get(&(holder as usize)).is_some_and(|i| self.entries.groups[*i].whole.is_some()) || self.own.contains(&(holder as usize)) {
            return;
        }
        // What cannot be compared later is refused when the trace is over,
        // unless the holder died with the run.
        if self.changed.contains(&(holder as usize)) || !ordered_ok(holder, kind) {
            let at = place();
            let g = self.group(holder, kind);
            g.late = true;
            g.late_at = at;
            return;
        }
        let items = snapshot(holder, kind);
        if items.len() > MAX_ITEMS {
            self.refused = Some(format!("it reads a {} of {} items whole", type_name(holder), size(holder, kind)));
            return;
        }
        self.count += items.len() + 1;
        let items = items.into_iter().map(|x| counted(x)).collect();
        self.group(holder, kind).whole = Some(Whole { items, at: place() });
    }

    unsafe fn length(&mut self, holder: *mut PyObject, kind: Holder) {
        if self.own.contains(&(holder as usize)) {
            return;
        }
        if self.changed.contains(&(holder as usize)) {
            let at = place();
            let g = self.group(holder, kind);
            g.late = true;
            g.late_at = at;
            return;
        }
        let n = size(holder, kind);
        let g = self.group(holder, kind);
        if g.len.is_none() && g.whole.is_none() {
            g.len = Some(n);
            self.count += 1;
        }
    }

    // The run asks something of the tensor `o`, or hands it to C code: its
    // key as it is now, once per tensor (see Keyed).
    unsafe fn keyed(&mut self, o: *mut PyObject) {
        if let Some(i) = self.asked_at.get(&(o as usize))
            && ffi::PyWeakref_GetObject(self.asked_of[*i].weak.as_ptr()) == o
        {
            return;
        }
        let weak = ffi::PyWeakref_NewRef(o, std::ptr::null_mut());
        if weak.is_null() {
            ffi::PyErr_Clear();
            self.refused = Some(format!("it asks about a {}, which takes no weak reference", type_name(o)));
            return;
        }
        self.asked_at.insert(o as usize, self.asked_of.len());
        self.asked_of.push(Asked { weak: Py::from_owned_ptr(Python::assume_attached(), weak), key: TensorKey::of(o), ours: false });
    }

    unsafe fn container(o: *mut PyObject) -> Option<(*mut PyObject, Holder)> {
        if ffi::PyDict_Check(o) != 0 {
            Some((o, Holder::Dict))
        } else if ffi::PyList_Check(o) != 0 {
            Some((o, Holder::List))
        } else if ffi::PySet_Check(o) != 0 {
            Some((o, Holder::Set))
        } else {
            // A view of a dict, or a class's namespace.
            let d = tr_view_dict(o);
            let d = if d.is_null() { tr_proxy_dict(o) } else { d };
            (!d.is_null()).then_some((d, Holder::Dict))
        }
    }

    // `bound`: `o` is what C code came by through something it was handed
    // (an item, the self of a builtin method), not an operand itself.
    unsafe fn used(&mut self, o: *mut PyObject, bound: bool) {
        if o.is_null() {
            return;
        }
        if !bound {
            self.walked.clear();
        }
        if let Some((h, kind)) = Self::container(o) {
            self.whole(h, kind);
            self.through(h, Some(kind));
        } else if fixed(o).is_some() {
            self.through(o, None);
        } else if ffi::Py_TYPE(o) == &raw mut ffi::PyDictProxy_Type {
            self.refused = Some("it reads the namespace of something that isn't a class".into());
        } else if !bound {
            self.used(tr_builtin_self(o), true);
        }
        self.handed(o);
    }

    // C code that is handed a tuple or a frozenset (`kind` None) or a
    // container reads through it: the list inside the tuple of a subscript,
    // the rows of a nested list that is made into a tensor, the tensors of
    // the list that torch.cat is handed, of a starred tuple, of a `**` dict. The tuples,
    // lists, dicts and sets that `o` holds are read whole as well, and what
    // those hold; a tensor `o` holds is handed as one handed directly is.
    // Each is walked once per handing (one may hold itself; by address, so
    // not once per trace: a tuple that died leaves its address to the
    // next), and no further once the trace has more entries than a region
    // may have. Anything else `o` holds is not looked at: an object of a
    // Python class answers C code by frames, which record what they read.
    unsafe fn through(&mut self, o: *mut PyObject, kind: Option<Holder>) {
        if self.count > MAX_ENTRIES || self.own.contains(&(o as usize)) || !self.walked.insert(o as usize) {
            return;
        }
        let items = match kind {
            None => fixed(o).unwrap_or_default(),
            // One that cannot be walked is refused by whole(), now or when the trace is over.
            Some(kind) if ordered_ok(o, kind) && size(o, kind) <= MAX_ITEMS as isize => snapshot(o, kind),
            Some(_) => vec![],
        };
        for x in items {
            if tensor(x) {
                self.keyed(x);
            } else if ffi::PyTuple_Check(x) != 0 || ffi::PyList_Check(x) != 0 || ffi::PyDict_Check(x) != 0 || ffi::PyAnySet_Check(x) != 0 {
                self.used(x, true);
            }
        }
    }

    // C code was handed `o`: what it does with an object of a Python class
    // is what the class says (see special.rs). The object's class by
    // identity, and what the class has under each name C code resolves. A
    // tensor is asked (see keyed): its class is in its key, or the guard's;
    // a class that cannot change needs none.
    unsafe fn handed(&mut self, o: *mut PyObject) {
        let tp = ffi::Py_TYPE(o);
        if tensor(o) {
            self.keyed(o);
            return;
        }
        if tr_type_immutable(tp) != 0 {
            return;
        }
        if self.seen.insert((o as usize, 0)) {
            self.count += 1;
            self.entries.typed.push(Typed { obj: counted(o), class: counted(tp as *mut PyObject) });
        }
        if self.seen.insert((tp as usize, usize::MAX - 1)) {
            for name in special::names() {
                self.looked(tp, *name as *mut PyObject);
            }
        }
    }

    // obj's class decides what `name` gives: obj's class by identity, and
    // what the class has under that name.
    unsafe fn typed(&mut self, obj: *mut PyObject, name: *mut PyObject) {
        let tp = ffi::Py_TYPE(obj);
        if tr_type_immutable(tp) != 0 {
            return;
        }
        if self.seen.insert((obj as usize, 0)) {
            self.count += 1;
            self.entries.typed.push(Typed { obj: counted(obj), class: counted(tp as *mut PyObject) });
        }
        self.looked(tp, name);
    }

    // As typed, for a special method the interpreter looks up by its name.
    unsafe fn special(&mut self, obj: *mut PyObject, name: &std::ffi::CStr) {
        if !obj.is_null() && tr_type_heap(ffi::Py_TYPE(obj)) != 0 && ffi::PyType_Check(obj) == 0 {
            let name = ffi::PyUnicode_InternFromString(name.as_ptr());
            if !name.is_null() {
                self.typed(obj, name);
                ffi::Py_DECREF(name);
            }
        }
    }

    // What class tp has under `name` decides something.
    unsafe fn looked(&mut self, tp: *mut ffi::PyTypeObject, name: *mut PyObject) {
        if tr_type_immutable(tp) != 0 || !self.seen.insert((tp as usize, name as usize)) {
            return;
        }
        let i = match self.class_of.get(&(tp as usize)) {
            Some(i) => *i,
            None => {
                self.entries.classes.push(Class { class: counted(tp as *mut PyObject), tag: Cell::new(0), names: vec![] });
                self.class_of.insert(tp as usize, self.entries.classes.len() - 1);
                self.entries.classes.len() - 1
            }
        };
        self.count += 1;
        self.entries.classes[i].names.push((own(name), None));
    }

    // The traced run asked a question about a class (see asked.rs): what the
    // answer depends on besides the operands themselves.
    unsafe fn asked(&mut self, q: asked::Question) {
        let class_name = ffi::PyUnicode_InternFromString(c"__class__".as_ptr());
        let getattribute = ffi::PyUnicode_InternFromString(c"__getattribute__".as_ptr());
        if class_name.is_null() || getattribute.is_null() {
            ffi::PyErr_Clear();
            self.refused = Some("a name could not be made".into());
            return;
        }
        let of = if q.instance { ffi::Py_TYPE(q.subject) } else { q.subject as *mut ffi::PyTypeObject };
        if !q.instance && ffi::PyType_Check(q.subject) == 0 {
            self.refused = Some(format!("it asks issubclass of a {}, which is no class", type_name(q.subject)));
        } else if q.instance && tensor(q.subject) {
            // Which class a tensor is of is in its key.
            self.keyed(q.subject);
            self.looked(of, class_name);
            self.looked(of, getattribute);
        } else if q.instance && !q.against.is_null() && asked::answers_class_itself(q.subject, of, class_name, getattribute) {
            self.refused = Some(format!("it asks isinstance of a {}, which answers .__class__ itself", type_name(q.subject)));
        } else if q.instance {
            self.typed(q.subject, class_name);
            self.looked(of, getattribute);
        }
        ffi::Py_DECREF(class_name);
        ffi::Py_DECREF(getattribute);
        if self.refused.is_some() || q.against.is_null() {
            return;
        }
        if tr_type_immutable(of) == 0 && self.seen.insert((of as usize, usize::MAX)) {
            self.count += 1;
            self.entries.mros.push(Mro { class: counted(of as *mut PyObject), mro: own((*of).tp_mro) });
        }
        let special = if q.instance { c"__instancecheck__" } else { c"__subclasscheck__" };
        let mut against = vec![];
        asked::members(q.against, &mut against);
        let mut abstract_ = q.abstract_;
        for c in against {
            if ffi::PyType_Check(c) == 0 {
                self.special(c, special);
                continue;
            }
            abstract_ |= asked::abstract_(c);
            // Which metaclass decides, and what it has as the method.
            let name = ffi::PyUnicode_InternFromString(special.as_ptr());
            if !name.is_null() {
                self.typed(c, name);
                ffi::Py_DECREF(name);
            }
        }
        if abstract_ {
            let now = asked::token();
            if self.entries.abc.is_some_and(|was| was != now) {
                self.refused = Some("a class was registered with an ABC while it was traced".into());
            }
            self.entries.abc = Some(now);
        }
    }

    // A frame of the function `f` starts under the traced body. Its
    // keyword-only defaults are a dict: read whole, as a call fills from it.
    unsafe fn function(&mut self, f: *mut PyObject) {
        if f.is_null() || ffi::PyFunction_Check(f) == 0 || !self.seen.insert((f as usize, 1)) {
            return;
        }
        let o = f as *mut ffi::PyFunctionObject;
        let (code, defaults, kwdefaults) = ((*o).func_code, (*o).func_defaults, (*o).func_kwdefaults);
        let held = |p: *mut PyObject| (!p.is_null()).then(|| counted(p));
        self.count += 1;
        self.entries.functions.push(Function { func: counted(f), code: counted(code), defaults: held(defaults), kwdefaults: held(kwdefaults) });
        if !kwdefaults.is_null() && ffi::PyDict_Check(kwdefaults) != 0 {
            self.whole(kwdefaults, Holder::Dict);
        }
    }

    // obj's attributes were looked up in `dict`, its __dict__.
    unsafe fn dicted(&mut self, obj: *mut PyObject, dict: *mut PyObject) {
        if !dict.is_null() && self.seen.insert((obj as usize, 2)) {
            self.count += 1;
            self.entries.dicted.push(Dicted { obj: counted(obj), dict: counted(dict) });
        }
    }

    // What a call of `obj` reads from it, for the two classes of C whose
    // objects a region may call: a functools.partial's function, arguments
    // and keywords (a dict: whole); and the cache of a function behind
    // functools.lru_cache, which nothing here can look at.
    unsafe fn state_called(&mut self, obj: *mut PyObject) {
        if obj.is_null() {
            return;
        }
        if ffi::PyObject_TypeCheck(obj, builtins().partial as *mut ffi::PyTypeObject) != 0 {
            for name in [c"func", c"args", c"keywords"] {
                let n = ffi::PyUnicode_InternFromString(name.as_ptr());
                let v = if n.is_null() { n } else { ffi::PyObject_GenericGetAttr(obj, n) };
                if v.is_null() {
                    ffi::PyErr_Clear();
                    self.refused = Some("it calls a functools.partial it cannot read".into());
                } else {
                    if self.seen.insert((obj as usize, n as usize)) {
                        self.count += 1;
                        self.entries.reads.push(Read { obj: counted(obj), name: Some(own(n)), want: Want::Is, value: Some(counted(v)), at: place() });
                    }
                    if ffi::PyDict_Check(v) != 0 {
                        self.whole(v, Holder::Dict);
                    }
                    ffi::Py_DECREF(v);
                }
                ffi::Py_XDECREF(n);
            }
        } else if super::effects::lru_cache_wrapper(obj) && !super::user_pure(obj) {
            self.refused = Some("it calls a function behind functools.lru_cache, whose cache it cannot check".into());
        }
    }

    // An attribute lookup on an instance of tp: which code resolves it.
    unsafe fn resolves(&mut self, obj: *mut PyObject) {
        self.special(obj, c"__getattribute__");
        self.special(obj, c"__getattr__");
    }

    // `generic`: the load was object.__getattribute__'s own (called on obj
    // directly), so it is resolved as that does, whatever obj's class does
    // with a load of its own.
    unsafe fn attr(&mut self, obj: *mut PyObject, name: *mut PyObject, value: *mut PyObject, ran: bool, generic: bool) {
        // A bound method is bound to the same function and object for as long
        // as it lives; its other attributes are its function's.
        if ffi::Py_TYPE(obj) == &raw mut super::ffi::PyMethod_Type {
            let n = text(name);
            if n != "__func__" && n != "__self__" {
                let func = ffi::PyObject_GetAttrString(obj, c"__func__".as_ptr());
                if func.is_null() {
                    ffi::PyErr_Clear();
                } else {
                    self.attr(func, name, value, ran, false);
                    ffi::Py_DECREF(func);
                }
            }
            return;
        }
        if tensor(obj) {
            // What a tensor's key names (exec/tensor_key.rs) is the guard's
            // to compare for an argument and the key's of a tensor the run
            // read from state (see Keyed); its values are the graph's.
            // Nothing else it can be asked is: an attribute the program put
            // on the object, or one of UNCHECKED_OF_A_TENSOR.
            self.keyed(obj);
            let n = text(name);
            let mut dict = std::ptr::null_mut();
            let own_attr = tr_attr_home(obj, name, &mut dict, 1) == 2;
            ffi::Py_XDECREF(dict);
            if own_attr || UNCHECKED_OF_A_TENSOR.contains(&n.as_str()) {
                self.refused = Some(format!("it reads a tensor's .{n}"));
            } else {
                // What the class has under the name (a method the program
                // put on torch.Tensor); which class it is, its key says.
                self.looked(ffi::Py_TYPE(obj), name);
            }
            return;
        }
        let mut dict = std::ptr::null_mut();
        let home = tr_attr_home(obj, name, &mut dict, generic as std::ffi::c_int);
        match home {
            0 => {}
            // A method, a class attribute, the class's __getattr__: the class,
            // and the instance not having it itself.
            1 => {
                self.typed(obj, name);
                self.resolves(obj);
                self.dicted(obj, dict);
                if !dict.is_null() {
                    self.item(dict, Holder::Dict, name, Want::Absent, std::ptr::null_mut());
                }
            }
            2 => {
                self.typed(obj, name);
                self.resolves(obj);
                self.dicted(obj, dict);
                self.dict_item(dict, name, false);
            }
            3 => {
                self.typed(obj, name);
                self.resolves(obj);
                let n = text(name);
                // `__dict__` too: the items read from the dict it gave are
                // obj's for as long as obj has that dict.
                let written = self.written.contains(&(obj as usize, name as usize));
                if !ran && !written && !value.is_null() && n != "__class__" && self.seen.insert((obj as usize, name as usize)) {
                    self.count += 1;
                    self.entries.reads.push(Read { obj: counted(obj), name: Some(own(name)), want: want_of(value), value: Some(counted(value)), at: place() });
                }
            }
            4 => {
                if ffi::Py_TYPE(obj) != &raw mut ffi::PyModule_Type {
                    self.typed(obj, name);
                }
                self.dict_item(dict, name, false);
            }
            5 => self.looked(obj as *mut ffi::PyTypeObject, name),
            _ => {
                if ran {
                    self.typed(obj, name);
                    self.resolves(obj);
                } else {
                    // Whatever becomes of obj: a class that resolves loads
                    // its own way in C may hand on what another object holds
                    // (a weak proxy does), which no entry would cover.
                    self.refused = Some(format!("it reads .{} of a {}, which it cannot check", text(name), type_name(obj)));
                }
            }
        }
        ffi::Py_XDECREF(dict);
    }
}

// The builtins whose reads are recorded by item rather than whole.
struct Builtins {
    partial: usize,
    getattr: usize,
    hasattr: usize,
    getattribute: usize,
    len: usize,
    dict_get: usize,
    dict_getitem: usize,
    dict_contains: usize,
}

fn builtins() -> &'static Builtins {
    static B: std::sync::OnceLock<Builtins> = std::sync::OnceLock::new();
    B.get_or_init(|| {
        let py = unsafe { Python::assume_attached() };
        let at = |code: &std::ffi::CStr| py.eval(code, None, None).map_or(0, |o| o.as_ptr() as usize);
        Builtins {
            partial: at(c"__import__('functools').partial"),
            getattr: at(c"getattr"),
            hasattr: at(c"hasattr"),
            getattribute: at(c"object.__getattribute__"),
            len: at(c"len"),
            dict_get: at(c"dict.__dict__['get']"),
            dict_getitem: at(c"dict.__dict__['__getitem__']"),
            dict_contains: at(c"dict.__dict__['__contains__']"),
        }
    })
}

// ---- For writes.rs: what the run's own writes mean for its reads -----------

// The traced run is about to store or delete item `key` of `holder` (a dict,
// a list, or for a slot the object itself): what it reads of that item from
// here on is what it wrote, and no entry.
pub(super) fn wrote_item(holder: *mut PyObject, key: *mut PyObject) {
    with(|t| {
        t.written.insert((holder as usize, key as usize));
    });
}

// The traced run is about to change `container` (an item of it, or its
// length): read whole or measured from here on, it is not what a later call
// finds (see Group::late).
pub(super) fn changed(container: *mut PyObject) {
    with(|t| {
        t.changed.insert(container as usize);
    });
}

// A deletion of `d[key]` raises unless the key is there: that it is, is a
// read.
pub(super) unsafe fn needs_key(d: *mut PyObject, key: *mut PyObject) {
    with(|t| t.dict_item(d, key, true));
}

// A store by index is in range for a list as long as this one: its length is
// a read.
pub(super) unsafe fn needs_length(list: *mut PyObject) {
    with(|t| t.length(list, Holder::List));
}

// Drops the entries of the running trace that only its own records keep
// alive (see sweep).
pub(super) fn sweep_trace() {
    TRACE.with_borrow_mut(|t| {
        if let Some(t) = t {
            unsafe { sweep(&mut t.entries) };
            t.group_of = t.entries.groups.iter().enumerate().map(|(i, g)| (g.holder.as_ptr() as usize, i)).collect();
        }
    });
}

// Every reference the running trace's entries hold, reported to `hold`.
pub(super) fn held_by_trace(hold: &mut impl FnMut(&Py<PyAny>)) {
    TRACE.with_borrow(|t| {
        if let Some(t) = t {
            held(&t.entries, hold);
        }
    });
}

pub(super) fn owned(container: *mut PyObject) -> bool {
    TRACE.with_borrow(|t| t.as_ref().is_some_and(|t| t.own.contains(&(container as usize))))
}

/// These containers are the running trace's own (they live as long as it
/// does): what the body's run reads from them or writes to them is neither a
/// read nor a write of the body.
#[pyfunction]
pub fn _region_own(objs: Vec<Bound<'_, PyAny>>) {
    TRACE.with_borrow_mut(|t| {
        if let Some(t) = t {
            t.own.extend(objs.iter().map(|o| o.as_ptr() as usize));
        }
    });
}

// A global load in a frame with these globals and builtins gave `value`.
pub unsafe fn global(globals: *mut PyObject, builtins: *mut PyObject, name: *mut PyObject, value: *mut PyObject) {
    with(|t| {
        if ffi::PyDict_Check(globals) == 0 || ffi::PyDict_Check(builtins) == 0 {
            t.refused = Some("it loads a global from a namespace that isn't a dict".into());
            return;
        }
        if dict_get(globals, name).is_null() {
            t.item(globals, Holder::Dict, name, Want::Absent, std::ptr::null_mut());
            t.item(builtins, Holder::Dict, name, want_of(value), value);
            t.group(builtins, Holder::Dict).root = true;
        } else {
            t.item(globals, Holder::Dict, name, want_of(value), value);
        }
        t.group(globals, Holder::Dict).root = true;
    });
}

// obj.name was loaded: `value` (NULL: it isn't there), `ran` if that entered
// a Python frame (a property, __getattr__), whose own reads are recorded.
// The exception set, if any, is kept.
pub unsafe fn attr(obj: *mut PyObject, name: *mut PyObject, value: *mut PyObject, ran: bool) {
    let raised = if value.is_null() { ffi::PyErr_GetRaisedException() } else { std::ptr::null_mut() };
    with(|t| t.attr(obj, name, value, ran, false));
    if !raised.is_null() {
        ffi::PyErr_SetRaisedException(raised);
    }
}

// What obj's class has under `name` decided what ran (a guard's check).
pub unsafe fn class_of(obj: *mut PyObject, name: *mut PyObject) {
    with(|t| t.typed(obj, name));
}

// super().name was loaded in a method of class `cls` on `obj`: what the
// class after `cls` in the order obj's class resolves methods in has.
pub unsafe fn super_attr(cls: *mut PyObject, obj: *mut PyObject, name: *mut PyObject) {
    with(|t| {
        t.typed(obj, name);
        let next = tr_super_next(cls, obj);
        if !next.is_null() {
            t.looked(next as *mut ffi::PyTypeObject, name);
        }
    });
}

// A frame of the Python function `f` starts under the traced body.
pub unsafe fn function(f: *mut PyObject) {
    with(|t| t.function(f));
}

// obj, which is no function, was called: its class's __call__ ran, and if
// its class is C's, what that reads from the object.
pub unsafe fn called(obj: *mut PyObject) {
    with(|t| {
        t.special(obj, c"__call__");
        t.state_called(obj);
    });
}

// A load is about to ask the defaultdict `container`, which outlives the
// run, for a key it does not have: C code stores what the factory makes.
// The load's item is then recorded as it is afterwards, whether or not the
// factory ran a frame; and where the region checks the dict itself, as it
// is handed it (no entry of the reads stands for it), the trace is refused
// when it is over (see _region_reads).
pub unsafe fn fills(container: *mut PyObject) {
    with(|t| {
        t.filled.entry(container as usize).or_insert_with(|| place());
    });
}

// container[key] was read (`ran`: by Python code, which records its own).
pub unsafe fn subscript(container: *mut PyObject, key: *mut PyObject, ran: bool) {
    with(|t| {
        if ffi::PyDict_Check(container) != 0 && key_ok(key) {
            // A dict subclass's __missing__ ran because C code found no such
            // key: no frame reads that, and one that stores nothing leaves
            // the key absent. Nor does the frame of a defaultdict's factory
            // read that the key was missing.
            if !ran || dict_get(container, key).is_null() || t.filled.contains_key(&(container as usize)) {
                t.dict_item(container, key, false);
            }
        } else if ffi::PyList_Check(container) != 0 && ffi::Py_TYPE(key) == &raw mut ffi::PyLong_Type && !ran {
            let i = ffi::PyLong_AsSsize_t(key);
            if i >= 0 && i < ffi::PyList_GET_SIZE(container) {
                t.item(container, Holder::List, key, want_of(ffi::PyList_GET_ITEM(container, i)), ffi::PyList_GET_ITEM(container, i));
            } else {
                // From the end: the length is part of it.
                ffi::PyErr_Clear();
                t.whole(container, Holder::List);
            }
        } else {
            t.used(container, false);
            t.used(key, false);
            t.special(container, c"__getitem__");
        }
    });
}

// `key in container` was evaluated.
pub unsafe fn contains(container: *mut PyObject, key: *mut PyObject, ran: bool) {
    with(|t| {
        if ffi::PyDict_Check(container) != 0 && key_ok(key) {
            if !ran {
                t.dict_item(container, key, true);
            }
        } else {
            t.used(container, false);
            t.special(container, c"__contains__");
        }
    });
}

// obj was iterated.
pub unsafe fn iterated(obj: *mut PyObject) {
    with(|t| {
        t.used(obj, false);
        t.special(obj, c"__iter__");
    });
}

// A cell's content was read.
pub unsafe fn cell(cell: *mut PyObject, value: *mut PyObject) {
    with(|t| {
        if t.seen.insert((cell as usize, 0)) {
            t.count += 1;
            t.entries.reads.push(Read { obj: counted(cell), name: None, want: want_of(value), value: Some(counted(value)), at: String::new() });
        }
    });
}

// An instruction that ran C code was handed these: the containers among them
// are read whole.
pub unsafe fn used(objs: &[*mut PyObject]) {
    with(|t| objs.iter().for_each(|o| t.used(*o, false)));
}

// `f(*o)` ran with a Python callee. A lone starred operand that is no tuple
// is made into the frame's tuple by the instruction itself, in C: a container
// is read whole, in order (its items are handed on unread), and an object of
// a Python class is iterated as its class says. A tuple is the frame's as it
// is: nothing is read.
pub unsafe fn starred(o: *mut PyObject) {
    if o.is_null() || ffi::PyTuple_CheckExact(o) != 0 {
        return;
    }
    with(|t| {
        if let Some((h, kind)) = Trace::container(o) {
            t.whole(h, kind);
        }
        t.handed(o);
    });
}

// The truth (or length) of obj was taken.
pub unsafe fn truth(obj: *mut PyObject) {
    with(|t| {
        if let Some((h, kind)) = Trace::container(obj).filter(|(h, _)| *h == obj) {
            t.length(h, kind);
        } else {
            if tensor(obj) {
                t.keyed(obj);
            }
            t.special(obj, c"__bool__");
            t.special(obj, c"__len__");
        }
    });
}

// `callee` (a builtin: a C function, a method of a builtin type, a builtin
// type) was called with these positional arguments (then keyword values)
// and returned `result` (null: it raised; the exception set is kept).
pub unsafe fn call(callee: *mut PyObject, args: &[*mut PyObject], result: *mut PyObject, ran: bool) {
    let raised = if result.is_null() { ffi::PyErr_GetRaisedException() } else { std::ptr::null_mut() };
    call_read(callee, args, result, ran);
    built(callee, args, result);
    if !raised.is_null() {
        ffi::PyErr_SetRaisedException(raised);
    }
}

unsafe fn call_read(callee: *mut PyObject, args: &[*mut PyObject], result: *mut PyObject, ran: bool) {
    let b = builtins();
    let f = tr_bound_function(callee) as usize;
    let me = tr_builtin_self(callee);
    let mut all = vec![];
    if !me.is_null() {
        all.push(me);
    }
    all.extend_from_slice(args);
    let str_at = |k: usize| all.len() > k && ffi::Py_TYPE(all[k]) == &raw mut ffi::PyUnicode_Type;
    if (f == b.getattr || f == b.getattribute || f == b.hasattr) && str_at(1) {
        let value = if f == b.hasattr { std::ptr::null_mut() } else { result };
        with(|t| t.attr(all[0], all[1], value, ran, f == b.getattribute));
    } else if (f == b.dict_get || f == b.dict_getitem || f == b.dict_contains) && all.len() >= 2 && ffi::PyDict_Check(all[0]) != 0 && key_ok(all[1]) {
        with(|t| t.dict_item(all[0], all[1], f == b.dict_contains));
    } else if f == b.len && all.len() == 1 {
        truth(all[0]);
    } else {
        if let Some(q) = asked::question(f, &all).or_else(|| asked::question(callee as usize, &all)) {
            with(|t| t.asked(q));
        }
        used(&all);
    }
}

// `callee`, which is no Python function, was handed `args` (positional, then
// keyword values) and returned `result` (NULL: it raised) to the traced
// run: what it built that the region's graph does not make refuses the
// trace (built.rs).
pub unsafe fn built(callee: *mut PyObject, args: &[*mut PyObject], result: *mut PyObject) {
    if !result.is_null()
        && let Some(why) = super::built::refusal(callee, args, result)
    {
        refuse(why);
    }
}

// As built(), for a starred call: `args` is the starred operand and
// `kwargs` the dict (or NULL).
pub unsafe fn built_starred(callee: *mut PyObject, args: *mut PyObject, kwargs: *mut PyObject, result: *mut PyObject) {
    built(callee, &super::built::spread(args, kwargs), result);
}

// A tensor the region's graph reads that is none of its arguments: a weight.
// The region holds the object (the read of its home says it is still the
// one) and is entered while it has the metadata of the trace, which the
// graph is specialized on. Where its data is may change under the same
// object (`param.data = ...`): the graph reads the tensor wherever it is,
// and a lowering that was bound to the constants is bound anew first.
struct Constant {
    obj: Py<PyAny>,
    // What it was at the trace (the class too: the program can bind another
    // to the same object).
    key: TensorKey,
    at: Cell<usize>,
    // For a constant a pass keeps something derived from (torch.vm.Unwritten):
    // its version counter when the pass looked. The region is entered while
    // the counter is that and the storage's copy-on-write mark is still set.
    unwritten: Option<i64>,
}

// What a region is checked against when it is entered (see the top of the
// file). Made by torch._rust._region_reads when the body's trace is over.
#[pyclass(frozen, module = "torch._rust")]
pub struct Reads {
    thread: u64,
    entries: GilCell<Entries>,
    consts: GilCell<Vec<Constant>>,
    // Called before the region's function when a constant's data moved (set
    // for a lowering bound to the constants), and whether one did.
    rebind: GilCell<Option<Py<PyAny>>>,
    moved: Cell<bool>,
    // How the region's tensors share memory, which is told when a constant's
    // data moved (see exec/overlap.rs).
    overlap: GilCell<Option<Py<Overlap>>>,
    // An entry's holder was found dead at the last check: such entries are
    // dropped at the next.
    lost: Cell<bool>,
    // An object the entries expect to find died: no check passes again.
    dead: Cell<bool>,
    misses: Cell<u64>,
    last: Cell<&'static str>,
    // The group the last failed check failed on, looked at first: a region
    // traced for a state that is gone fails there again.
    hot: Cell<usize>,
}

unsafe impl Sync for Reads {}

// Drops what was read from an object that nothing but these entries refers
// to any more: an object the traced run made (or let go of), which no later
// call can reach, so there is nothing to check. What was read from it was
// its own (its __dict__, its slots: the entries here are of loads resolved
// from the object itself or from its class, and the class's stay). An object
// is such when every reference to it is one of the entries' own (as what was
// read from, or as a value read). Repeated: an object dropped may have been
// all that held another (an instance and its __dict__, a temporary and what
// it was made of).
unsafe fn sweep(e: &mut Entries) {
    for _ in 0..SWEEP_ROUNDS {
        let mut mine: HashMap<usize, isize> = HashMap::new();
        held(e, &mut |o: &Py<PyAny>| *mine.entry(o.as_ptr() as usize).or_insert(0) += 1);
        // Judged before any entry is dropped: dropping one lets go of what
        // it holds, and the counts above would no longer be the entries'
        // (an object with as many other owners as entries dropped before it
        // would pass for gone, and the checks on it would be lost).
        let gone: HashSet<usize> = mine.iter().filter(|(o, n)| ffi::Py_REFCNT(**o as *mut PyObject) == **n).map(|(o, _)| *o).collect();
        let gone = |o: &Pin| gone.contains(&(o.as_ptr() as usize));
        let before = (e.typed.len(), e.reads.len(), e.groups.len(), e.functions.len() + e.dicted.len());
        e.typed.retain(|t| !gone(&t.obj));
        e.reads.retain(|r| !gone(&r.obj));
        e.groups.retain(|g| !gone(&g.holder) && !g.empty());
        e.functions.retain(|f| !gone(&f.func));
        e.dicted.retain(|d| !gone(&d.obj));
        if before == (e.typed.len(), e.reads.len(), e.groups.len(), e.functions.len() + e.dicted.len()) {
            break;
        }
    }
    unfound(e);
}

// Where a walk of the reads starts (see led).
#[derive(Clone, Copy)]
enum Start<'a> {
    // From what was there before the traced run: the namespaces globals
    // were loaded from, the cells the run read and neither made nor stored
    // into, and these (the region's arguments). That an entry was read FROM
    // an object leads nowhere: a run reads what it made too (made.rs).
    Before(&'a [*mut PyObject]),
    // From every entry, whatever it was read from: what the entries hold
    // the program to, for as long as they are entries.
    Entries,
}

// The objects the reads `e` lead to, by address: the one answer to "do the
// reads lead to this object", for a container a region hands back
// (Reads.knows, _region_known), a tensor the run asked about (Keyed) and an
// input the run also read (Reads.values). An object is led to from where
// the walk starts by what the entries expect by identity, and only by that:
//   - an item of a namespace or of a container that is led to, read alone
//     or with the whole content;
//   - an attribute of an object that is led to: an item of its __dict__, a
//     slot or another attribute C code gives, what its class or a base has
//     under a name;
//   - the items of a tuple or a frozenset, at any depth; what a bound
//     method (of a Python function or of C code) is bound to; the dict a
//     view of a dict (keys, values, items) or a mappingproxy shows: none of
//     them can be bound anew;
//   - the defaults and the keyword-only defaults of a function a frame of
//     which ran (its Function entry says they are still the ones).
// Nothing else leads anywhere: an object some C code handed out, an item of
// a container no entry is of, a default of a function that did not run.
unsafe fn led(e: &Entries, start: Start) -> HashSet<usize> {
    let mut known: HashSet<usize> = HashSet::new();
    let (mut fresh, all) = match start {
        Start::Before(given) => (given.to_vec(), false),
        Start::Entries => (vec![], true),
    };
    // (NULL for an object that died: the walk passes over it.)
    let is = |v: &Option<Pin>| v.as_ref().map(|v| v.as_ptr());
    loop {
        // What an object that is led to brings with it.
        while let Some(o) = fresh.pop() {
            if o.is_null() || !known.insert(o as usize) {
                continue;
            }
            if let Some(items) = fixed(o) {
                fresh.extend(items);
            } else if !tr_view_dict(o).is_null() || !tr_proxy_dict(o).is_null() {
                // A view of a dict and a mappingproxy show the one dict they
                // were made of for as long as they live.
                fresh.push(if tr_view_dict(o).is_null() { tr_proxy_dict(o) } else { tr_view_dict(o) });
            } else if !tr_bound_self(o).is_null() {
                fresh.push(tr_bound_self(o));
            } else if !tr_builtin_self(o).is_null() {
                fresh.push(tr_builtin_self(o));
            } else if !all && ffi::PyFunction_Check(o) == 0 && ffi::PyList_Check(o) == 0 && ffi::PyDict_Check(o) == 0 && ffi::PyAnySet_Check(o) == 0 {
                // Its __dict__, whose items are entries of their own (from
                // every entry, those start the walk already).
                fresh.push(instance_dict(o));
            }
            // Its class and the bases were there before it.
            let class = if ffi::PyType_Check(o) != 0 { o as *mut ffi::PyTypeObject } else { ffi::Py_TYPE(o) };
            let mro = (*class).tp_mro;
            if !mro.is_null() && known.insert(class as usize) {
                known.extend((0..ffi::PyTuple_GET_SIZE(mro)).map(|i| ffi::PyTuple_GET_ITEM(mro, i) as usize));
            }
        }
        // What was read from an object that died (or from the __dict__ of
        // an owner that has none now) leads nowhere, wherever the walk
        // starts: no later call comes by it, and the entry is dropped at the
        // next check.
        let alive = |o: &Pin| o.live().filter(|o| !o.is_null());
        let led = |o: &Pin| alive(o).is_some_and(|o| all || known.contains(&(o as usize)));
        for g in e.groups.iter().filter(|g| g.root || led(&g.holder)) {
            fresh.extend(g.items.iter().filter(|i| i.want == Want::Is).filter_map(|i| is(&i.value)));
            fresh.extend(g.whole.iter().flat_map(|w| w.items.iter().map(|x| x.as_ptr())));
        }
        for r in e.reads.iter().filter(|r| r.want == Want::Is) {
            let from = if r.name.is_none() { alive(&r.obj).is_some_and(|c| all || !super::made::own_cell(c)) } else { led(&r.obj) };
            fresh.extend(is(&r.value).filter(|_| from));
        }
        for c in e.classes.iter().filter(|c| led(&c.class)) {
            fresh.extend(c.names.iter().filter_map(|(_, had)| is(had)));
        }
        for f in e.functions.iter().filter(|f| led(&f.func)) {
            fresh.extend(is(&f.defaults).into_iter().chain(is(&f.kwdefaults)));
        }
        fresh.retain(|o| !o.is_null() && !known.contains(&(*o as usize)));
        if fresh.is_empty() {
            return known;
        }
    }
}

// Drops the key of a tensor no entry leads to any more: the read that found
// it is gone (its holder died, or is the region's own to check), and with it
// the way a later call would come by the tensor. Returns whether one of the
// tensors dropped is neither an input of the trace nor a result of it.
unsafe fn unfound(e: &mut Entries) -> bool {
    if e.keyed.is_empty() {
        return false;
    }
    let led = led(e, Start::Entries);
    let before = e.keyed.iter().filter(|k| !k.ours).count();
    e.keyed.retain(|k| led.contains(&(k.obj.as_ptr() as usize)));
    e.keyed.iter().filter(|k| !k.ours).count() != before
}

// Every reference the entries count, reported to `hold`.
fn held(e: &Entries, hold: &mut impl FnMut(&Py<PyAny>)) {
    let mut pin = |p: &Pin| p.counts().into_iter().for_each(&mut *hold);
    for t in &e.typed {
        pin(&t.obj);
        pin(&t.class);
    }
    for c in &e.classes {
        pin(&c.class);
        c.names.iter().filter_map(|(_, had)| had.as_ref()).for_each(&mut pin);
    }
    for r in &e.reads {
        pin(&r.obj);
        r.value.iter().for_each(&mut pin);
    }
    for f in &e.functions {
        pin(&f.func);
        pin(&f.code);
        f.defaults.iter().chain(&f.kwdefaults).for_each(&mut pin);
    }
    for d in &e.dicted {
        pin(&d.obj);
        pin(&d.dict);
    }
    e.mros.iter().for_each(|m| pin(&m.class));
    e.keyed.iter().for_each(|k| pin(&k.obj));
    for g in &e.groups {
        pin(&g.holder);
        g.items.iter().filter_map(|i| i.value.as_ref()).for_each(&mut pin);
        g.whole.iter().flat_map(|w| &w.items).for_each(&mut pin);
    }
    for c in &e.classes {
        c.names.iter().for_each(|(n, _)| hold(n));
    }
    e.reads.iter().filter_map(|r| r.name.as_ref()).for_each(&mut *hold);
    e.mros.iter().for_each(|m| hold(&m.mro));
    e.groups.iter().flat_map(|g| &g.items).for_each(|i| hold(&i.key));
}

// The trace is over: from now on the entries hold by pin.rs, so that the
// region keeps alive nothing the program drops. By a weak reference: an
// instance, a module, a function, a code object, a class, a tensor, a set, a
// bound method, an instance of a subclass of dict or list. An instance's (or
// a module's) __dict__: through the instance, and the items read are then
// compared in the dict the instance has when the region is entered, which
// must be the dict it had: the entry that says which dict the instance has
// stays, by the dict's address (Pin::at), because something else may still
// read the dict that was read (a view or a mappingproxy made of it, kept
// somewhere) after the instance was given an equal one. Only
// where the instance is the one way the trace came by the dict: a dict that
// outlives its instance and is found some other way must stay checked when
// the instance died. Still counted, for as long as the region lives,
// because the class takes no weak reference and nothing here knows an owner
// to find it through:
//   - an int, a float, a str, a tuple, None: nothing mutable the program
//     could see live on, except what a tuple holds;
//   - an exact dict or list that is no instance's __dict__ (a dict found as
//     a value: nn.Module's _parameters, _modules and _buffers; a function's
//     keyword defaults), and with it what it holds;
//   - a __dict__ that is also a namespace a global was loaded from, that
//     two instances have, or that an entry expects to find as a value (what
//     `obj.__dict__` gave);
//   - a cell;
//   - an instance of a class with __slots__ and no __weakref__, and the
//     __dict__ of such an instance;
//   - a property, a staticmethod, a slot descriptor and the like a class
//     has under a name (dropped with the entry when the class died).
unsafe fn weaken(e: &mut Entries) {
    for t in &mut e.typed {
        t.obj.weaken();
        t.class.weaken();
    }
    // (Before anything is weakened: every pin is of a live object.)
    let found = led(e, Start::Entries);
    let dicts: Vec<usize> = e.dicted.iter().map(|d| d.dict.as_ptr() as usize).collect();
    let alone = |dict: usize| !found.contains(&dict) && dicts.iter().filter(|d| **d == dict).count() == 1;
    for d in &mut e.dicted {
        d.obj.weaken();
    }
    let groups = &mut e.groups;
    for d in &mut e.dicted {
        let dict = d.dict.as_ptr();
        let group = groups.iter_mut().find(|g| g.holder.is(dict));
        // (A namespace a global was loaded from stays counted.)
        if alone(dict as usize) && !group.as_ref().is_some_and(|g| g.root) && d.dict.at(&d.obj) && let Some(g) = group {
            g.holder.through(&d.obj);
        }
    }
    for g in groups.iter_mut() {
        g.holder.weaken();
        g.items.iter_mut().filter(|i| i.want == Want::Is).filter_map(|i| i.value.as_mut()).for_each(Pin::weaken);
        g.whole.iter_mut().flat_map(|w| &mut w.items).for_each(Pin::weaken);
    }
    for c in &mut e.classes {
        c.class.weaken();
        c.names.iter_mut().filter_map(|(_, had)| had.as_mut()).for_each(Pin::weaken);
    }
    for r in &mut e.reads {
        r.obj.weaken();
        r.value.iter_mut().filter(|_| r.want == Want::Is).for_each(Pin::weaken);
    }
    for f in &mut e.functions {
        f.func.weaken();
        f.code.weaken();
        f.defaults.iter_mut().chain(&mut f.kwdefaults).for_each(Pin::weaken);
    }
    e.mros.iter_mut().for_each(|m| m.class.weaken());
    e.keyed.iter_mut().for_each(|k| k.obj.weaken());
}

// Takes out the entries whose holder died, for the caller to drop.
fn lost(e: &mut Entries) -> Entries {
    fn take<T>(all: &mut Vec<T>, dead: impl Fn(&T) -> bool) -> Vec<T> {
        let (gone, kept) = std::mem::take(all).into_iter().partition(|x| dead(x));
        *all = kept;
        gone
    }
    Entries {
        groups: take(&mut e.groups, |g| g.holder.dead()),
        typed: take(&mut e.typed, |t| t.obj.dead()),
        classes: take(&mut e.classes, |c| c.class.dead()),
        reads: take(&mut e.reads, |r| r.obj.dead()),
        functions: take(&mut e.functions, |f| f.func.dead()),
        dicted: take(&mut e.dicted, |d| d.obj.dead()),
        mros: take(&mut e.mros, |m| m.class.dead()),
        abc: None,
        keyed: take(&mut e.keyed, |k| k.obj.dead()),
    }
}

// Whether an object the entries expect to find, in a holder that lives,
// died: it is found nowhere again.
fn expects_dead(e: &Entries) -> bool {
    let gone = |v: &Option<Pin>| v.as_ref().is_some_and(Pin::dead);
    e.groups.iter().any(|g| !g.holder.dead() && (g.items.iter().any(|i| gone(&i.value)) || g.whole.iter().flat_map(|w| &w.items).any(Pin::dead)))
        || e.typed.iter().any(|t| !t.obj.dead() && t.class.dead())
        || e.classes.iter().any(|c| !c.class.dead() && c.names.iter().any(|(_, had)| gone(had)))
        || e.reads.iter().any(|r| !r.obj.dead() && gone(&r.value))
        || e.functions.iter().any(|f| !f.func.dead() && (f.code.dead() || gone(&f.defaults) || gone(&f.kwdefaults)))
}

impl Reads {
    // What kind of read the last failed check failed on.
    pub fn last(&self) -> &'static str {
        self.last.get()
    }

    // Whether an object the entries expect to find died (found at a check
    // that failed): the region can never be entered again.
    pub fn dead(&self) -> bool {
        self.dead.get()
    }

    // Whether everything the trace read is as it was. No Python code runs
    // while the entries are compared (dropping the entries of a holder that
    // died, first, lets go of what they still count).
    pub unsafe fn check(&self) -> bool {
        if self.lost.replace(false) {
            let gone = lost(&mut *self.entries.get());
            self.hot.set(0);
            drop(gone);
        }
        let e = &*self.entries.get();
        // None: the holder died. No later call comes by it, so what was
        // read from it is nothing to check.
        let live = |p: &Pin| {
            let o = p.live();
            if o.is_none() {
                self.lost.set(true);
            }
            o
        };
        let passes = |g: &Group| {
            let Some(h) = live(&g.holder) else { return true };
            // (The owner of a __dict__ has none now.)
            if h.is_null() {
                return false;
            }
            // The order of an OrderedDict changes under the same version tag.
            let tagged = g.kind == Holder::Dict && !(g.whole.is_some() && tr_ordered(h) != 0);
            let v = if tagged { tr_dict_version(h) } else { 0 };
            if v != 0 && v == g.version.get() {
                return true;
            }
            let ok = g.holds();
            if ok {
                g.version.set(v);
            }
            ok
        };
        let why = if tffi::region_thread_state() != self.thread {
            "thread state"
        } else if tffi::torch_function_mode_active() {
            // The mode is to see the program's own calls: a region makes none.
            "a torch function mode"
        } else if e.groups.get(self.hot.get()).is_some_and(|g| !passes(g)) {
            "a namespace or a container"
        } else if e.typed.iter().any(|t| live(&t.obj).is_some_and(|o| !t.class.is(ffi::Py_TYPE(o) as *mut PyObject))) {
            "the class of an object"
        } else if !e.classes.iter().all(|c| {
            let Some(class) = live(&c.class) else { return true };
            let tp = class as *mut ffi::PyTypeObject;
            if tr_type_version_is(tp, c.tag.get()) != 0 {
                return true;
            }
            let ok = c.holds();
            if ok {
                c.tag.set(tr_type_version(tp));
            }
            ok
        }) {
            "a class"
        } else if !e.mros.iter().all(|m| live(&m.class).is_none() || m.holds()) {
            "the bases of a class"
        } else if e.abc.is_some_and(|was| asked::token() != was) {
            "a class registered with an ABC"
        } else if let Some(i) = e.groups.iter().position(|g| !passes(g)) {
            self.hot.set(i);
            "a namespace or a container"
        } else if !e.reads.iter().all(|r| live(&r.obj).is_none() || r.holds()) {
            "a cell or a slot"
        } else if !e.functions.iter().all(|f| live(&f.func).is_none() || f.holds()) {
            "the code or the defaults of a function"
        } else if e.dicted.iter().any(|d| live(&d.obj).is_some_and(|o| !d.dict.is(tr_instance_dict(o)))) {
            "the __dict__ of an object"
        } else if let Some(why) = self.constants_fail() {
            why
        } else if !e.keyed.iter().all(|k| live(&k.obj).is_none_or(|o| k.key.holds(o))) {
            // Behind the constants: a tensor that is both is a constant's miss.
            "the class, shape, dtype, device or requires_grad of a tensor read from state"
        } else {
            return true;
        };
        self.misses.set(self.misses.get() + 1);
        MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.last.set(why);
        if expects_dead(e) {
            self.dead.set(true);
        }
        false
    }

    // What a constant fails on, if one does: the metadata of the trace, or,
    // under torch.vm.Unwritten, not having been written to. For a lowering
    // bound to the constants, and for the check of how the region's tensors
    // share memory, also notes whether the data of one is somewhere else
    // than at the last look.
    unsafe fn constants_fail(&self) -> Option<&'static str> {
        let metadata = "the class, shape, dtype, device or requires_grad of a constant";
        let overlap = (*self.overlap.get()).as_ref();
        let watch = (*self.rebind.get()).is_some() || overlap.is_some();
        for c in &*self.consts.get() {
            let Some(t) = tffi::tensor_by_type(c.obj.as_ptr() as usize).as_ref() else { return Some(metadata) };
            if !c.key.holds(c.obj.as_ptr()) {
                return Some(metadata);
            }
            if let Some(version) = c.unwritten
                && (tffi::tensor_version(t) != version || !tffi::tensor_is_cow(t))
            {
                return Some("a constant that was to stay unwritten");
            }
            if watch {
                let at = tffi::tensor_data_address(t);
                if at != c.at.get() {
                    c.at.set(at);
                    self.moved.set(true);
                    overlap.iter().for_each(|o| o.get().moved());
                }
            }
        }
        None
    }

    // Has the lowering bound anew if a constant's data moved since it was
    // bound. Runs Python code: after the check, before the region's function.
    pub fn rebind(&self, py: Python<'_>) -> PyResult<()> {
        if self.moved.replace(false)
            && let Some(f) = unsafe { &*self.rebind.get() }
        {
            f.bind(py).call0()?;
        }
        Ok(())
    }
}

#[pymethods]
impl Reads {
    fn __len__(&self) -> usize {
        unsafe { &*self.entries.get() }.len()
    }

    /// How often the check failed, and on what kind of read the last time.
    #[getter]
    fn misses(&self) -> (u64, &'static str) {
        (self.misses.get(), self.last.get())
    }

    /// Adds ``holder[key] is value`` (a dict's item) unless that item is
    /// already checked.
    fn pin(&self, holder: &Bound<'_, PyAny>, key: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> bool {
        let e = unsafe { &mut *self.entries.get() };
        if unsafe { ffi::PyDict_Check(holder.as_ptr()) == 0 || !key_ok(key.as_ptr()) } {
            return false;
        }
        let i = match e.groups.iter().position(|g| g.holder.is(holder.as_ptr())) {
            Some(i) => i,
            None => {
                e.groups.push(Group { holder: unsafe { Pin::weak(holder.as_ptr()) }, kind: Holder::Dict, version: Cell::new(0), items: vec![], whole: None, len: None, late: false, late_at: String::new(), root: false });
                e.groups.len() - 1
            }
        };
        let g = &mut e.groups[i];
        let has = g.items.iter().any(|it| unsafe { exact::same(it.key.as_ptr(), key.as_ptr()) });
        let whole = g.whole.as_ref().is_some_and(|w| w.items.iter().any(|x| x.is(value.as_ptr())));
        if has || whole {
            return false;
        }
        g.items.push(Item { key: key.clone().unbind(), want: Want::Is, value: Some(unsafe { Pin::weak(value.as_ptr()) }) });
        g.version.set(0);
        true
    }

    /// The tensors the region's graph reads that are none of its arguments
    /// (weights): the region is entered while each has the key it has now
    /// (exec/tensor_key.rs). ``unwritten`` has, for a constant under
    /// torch.vm.Unwritten, the version counter it is to keep (None for the
    /// others). Returns how many of the constants are checked: those that
    /// have a key.
    fn constants(&self, consts: Vec<Bound<'_, PyAny>>, unwritten: Vec<Option<i64>>) -> usize {
        let list = unsafe { &mut *self.consts.get() };
        for (x, unwritten) in consts.into_iter().zip(unwritten) {
            let Some(key) = (unsafe { TensorKey::of(x.as_ptr()) }) else { continue };
            let at = unsafe { tffi::tensor_by_type(x.as_ptr() as usize).as_ref() }.map_or(0, tffi::tensor_data_address);
            list.push(Constant { obj: x.unbind(), key, at: Cell::new(at), unwritten });
        }
        list.len()
    }

    /// For a lowering bound to the constants: from now on ``rebind()`` is
    /// called, before the region's function, on an entry at which the data of
    /// a constant is not where it was at the entry before (or now).
    fn bound(&self, rebind: Bound<'_, PyAny>) {
        for c in unsafe { &*self.consts.get() } {
            if let Some(t) = unsafe { tffi::tensor_by_type(c.obj.as_ptr() as usize).as_ref() } {
                c.at.set(tffi::tensor_data_address(t));
            }
        }
        unsafe { *self.rebind.get() = Some(rebind.unbind()) };
        self.moved.set(false);
    }

    /// Which of ``objects``, by place, the traced run read as a value: from a
    /// namespace, an attribute, an item or a cell, or in a tuple it read so.
    /// (After ``forget``: what was read from the region's own arguments is
    /// no such read.)
    fn values(&self, objects: Vec<Bound<'_, PyAny>>) -> Vec<usize> {
        let read = unsafe { led(&*self.entries.get(), Start::Entries) };
        (0..objects.len()).filter(|&i| read.contains(&(objects[i].as_ptr() as usize))).collect()
    }

    /// For the check of how the region's tensors share memory: from now on
    /// it is told when the data of a constant is not where it was.
    fn shares(&self, overlap: Py<Overlap>) {
        overlap.get().moved();
        unsafe { *self.overlap.get() = Some(overlap) };
    }

    /// Whether what the trace read leads to ``obj`` from what was there
    /// before the traced run (see led; ``given``: the region's arguments): a
    /// region that holds an object itself (a list it hands back, which was
    /// there before it ran) has the right one for as long as its reads hold.
    fn knows(&self, obj: &Bound<'_, PyAny>, given: Vec<Bound<'_, PyAny>>) -> bool {
        let given: Vec<*mut PyObject> = given.iter().map(|g| g.as_ptr()).collect();
        unsafe { led(&*self.entries.get(), Start::Before(&given)).contains(&(obj.as_ptr() as usize)) }
    }

    /// Drops what was read from these containers and objects: the region
    /// checks them itself (its arguments, by structure), or makes them anew.
    /// Returns why the region cannot be, if it can't: a tensor the run asked
    /// about was found only by what is dropped, and is neither an input of
    /// the trace (the guard's to compare) nor a result of it.
    fn forget(&self, holders: Vec<Bound<'_, PyAny>>) -> Option<&'static str> {
        let e = unsafe { &mut *self.entries.get() };
        let given = |o: &Pin| holders.iter().any(|h| o.is(h.as_ptr()));
        e.groups.retain(|g| !given(&g.holder));
        e.typed.retain(|t| !given(&t.obj));
        e.reads.retain(|r| !given(&r.obj));
        e.dicted.retain(|d| !given(&d.obj));
        unsafe { unfound(e) }.then_some(UNLED)
    }

    /// The entries as text, for debugging.
    fn describe(&self, py: Python<'_>) -> Vec<String> {
        let e = unsafe { &*self.entries.get() };
        let shown = |o: &Py<PyAny>| {
            let o = o.bind(py);
            let tp = o.get_type().name().map_or("?".into(), |n| n.to_string());
            if unsafe { exact_scalar(o.as_ptr()) } || o.is_none() || o.is_instance_of::<pyo3::types::PyBool>() { format!("{}", o.repr().map_or("?".into(), |r| r.to_string())) } else { format!("<{tp} at {:p}>", o.as_ptr()) }
        };
        let show = |p: &Pin| unsafe { Bound::from_borrowed_ptr_or_opt(py, p.as_ptr()) }.map_or("<an object that died>".into(), |o| shown(&o.unbind()));
        let mut out = vec![format!("thread state {:#x}", self.thread)];
        out.extend(unsafe { &*self.consts.get() }.iter().map(|c| format!("{} stays a {}", shown(&c.obj), c.key)));
        out.extend(e.keyed.iter().map(|k| format!("{}, which was asked about, stays a {}", show(&k.obj), k.key)));
        out.extend(e.typed.iter().map(|t| format!("type({}) is {}", show(&t.obj), show(&t.class))));
        for c in &e.classes {
            let class = show(&c.class);
            out.extend(c.names.iter().map(|(n, had)| format!("class {class} has {} for {}", had.as_ref().map_or("nothing".into(), &show), n.bind(py))));
        }
        for g in &e.groups {
            let h = show(&g.holder);
            if let Some(n) = g.len {
                out.push(format!("len({h}) == {n}"));
            }
            if let Some(w) = &g.whole {
                out.push(format!("{h} holds the same {} objects{}", w.items.len(), w.at));
            }
            out.extend(g.items.iter().map(|i| match i.want {
                Want::Absent => format!("{} not in {h}", shown(&i.key)),
                Want::Present => format!("{} in {h}", shown(&i.key)),
                _ => format!("{h}[{}] {} {}", shown(&i.key), if i.want == Want::Is { "is" } else { "==" }, i.value.as_ref().map_or("?".into(), &show)),
            }));
        }
        out.extend(e.functions.iter().map(|f| format!("{} keeps its code and its defaults", show(&f.func))));
        out.extend(e.dicted.iter().map(|d| format!("{}.__dict__ is {}", show(&d.obj), show(&d.dict))));
        out.extend(e.reads.iter().map(|r| match &r.name {
            None => format!("{} holds {}", show(&r.obj), r.value.as_ref().map_or("?".into(), &show)),
            Some(n) => format!("{}.{} {} {}{}", show(&r.obj), n.bind(py), if r.want == Want::Is { "is" } else { "==" }, r.value.as_ref().map_or("?".into(), &show), r.at),
        }));
        out
    }
}

// obj's __dict__ (borrowed), or NULL if it has none.
pub(super) unsafe fn instance_dict(obj: *mut PyObject) -> *mut PyObject {
    let d = ffi::PyObject_GenericGetDict(obj, std::ptr::null_mut());
    if d.is_null() {
        ffi::PyErr_Clear();
        return d;
    }
    // The object holds it.
    ffi::Py_DECREF(d);
    d
}

/// Whether the run of the trace on this thread made the list, dict or set
/// ``obj`` (see made.rs): a region builds it anew on every call.
#[pyfunction]
pub fn _region_made(obj: &Bound<'_, PyAny>) -> bool {
    super::made::is(obj.as_ptr())
}

/// Whether the trace running on this thread came by the container ``obj``
/// through what it read, as far as it has run, from something that was
/// there before it (see led; ``given``: the region's arguments): then
/// ``obj`` was there before the traced run.
#[pyfunction]
pub fn _region_known(obj: &Bound<'_, PyAny>, given: Vec<Bound<'_, PyAny>>) -> bool {
    let given: Vec<*mut PyObject> = given.iter().map(|g| g.as_ptr()).collect();
    TRACE.with_borrow(|t| t.as_ref().is_some_and(|t| unsafe { led(&t.entries, Start::Before(&given)).contains(&(obj.as_ptr() as usize)) }))
}

/// The body of the trace on this thread is over, and everything is still as
/// its run left it (the trace's stand-ins for the floats of the arguments
/// are where the body found them): notes every read that is no longer as it
/// was read, unless a write the run recorded explains it (the holder is one
/// the run wrote to, the cell one it made or stored into). `_region_reads`
/// refuses the trace for each of them whose holder outlives the run.
/// ``ours``: the tensors that are the trace's inputs or the results of the
/// operators it recorded, as far as they are alive; of the tensors the run
/// asked about, those are the guard's to compare or the graph's to make.
#[pyfunction]
pub fn _region_settled(ours: Vec<Bound<'_, PyAny>>) {
    TRACE.with_borrow_mut(|t| {
        let Some(t) = t else { return };
        for o in &ours {
            if let Some(i) = t.asked_at.get(&(o.as_ptr() as usize))
                && unsafe { ffi::PyWeakref_GetObject(t.asked_of[*i].weak.as_ptr()) } == o.as_ptr()
            {
                t.asked_of[*i].ours = true;
            }
        }
        let e = &t.entries;
        let wrote: HashSet<usize> = t.written.iter().map(|(h, _)| *h).chain(t.changed.iter().copied()).collect();
        let mut stale = vec![];
        unsafe {
            for r in e.reads.iter().filter(|r| r.name.is_none() && !r.holds() && !super::made::own_cell(r.obj.as_ptr())) {
                let was = shown(r.value.as_ref().map_or(std::ptr::null_mut(), |v| v.as_ptr()));
                stale.push((Stale::Cell, r.obj.as_ptr() as usize, format!("a cell that held {was} and holds {}", shown(PyCell_Get_borrowed(r.obj.as_ptr())))));
            }
            for f in e.functions.iter().filter(|f| !f.holds()) {
                stale.push((Stale::Function, f.func.as_ptr() as usize, format!("the code and the defaults of {}, which are others", text((*(f.func.as_ptr() as *mut ffi::PyFunctionObject)).func_qualname))));
            }
            for d in e.dicted.iter().filter(|d| tr_instance_dict(d.obj.as_ptr()) != d.dict.as_ptr()) {
                stale.push((Stale::Dict, d.obj.as_ptr() as usize, format!("the __dict__ of a {}, which is another dict", type_name(d.obj.as_ptr()))));
            }
            for g in e.groups.iter().filter(|g| !wrote.contains(&(g.holder.as_ptr() as usize))) {
                let (h, kind) = (g.holder.as_ptr(), g.kind);
                let of = type_name(h);
                let item = g.items.iter().find(|i| !i.holds(h, kind)).map(|i| {
                    let was = match i.want {
                        Want::Absent => "nothing".into(),
                        Want::Present => "something".into(),
                        Want::Is | Want::Equal => shown(i.value.as_ref().map_or(std::ptr::null_mut(), |v| v.as_ptr())),
                    };
                    let now = if kind == Holder::Dict { shown(dict_get(h, i.key.as_ptr())) } else { "another object".into() };
                    format!("item {} of a {of}, which was {was} and is {now}", shown(i.key.as_ptr()))
                });
                let whole = g.whole.as_ref().filter(|w| !w.holds(h, kind)).map(|w| format!("a {of} whole{}, which holds something else", w.at));
                let len = g.len.filter(|n| size(h, kind) != *n).map(|n| format!("the length of a {of}, which was {n} and is {}", size(h, kind)));
                if let Some(what) = item.or(whole).or(len) {
                    stale.push((Stale::Group, h as usize, what));
                }
            }
        }
        t.stale = stale;
    });
}

// Why a trace that asked about such a tensor is no region (see _region_reads).
const UNLED: &str = "it asks about a tensor that is none of its inputs, that no operator of its run made and that none of its reads leads to";

static MISSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// How many times a region wasn't entered because what its trace read had
/// changed, over all regions.
#[pyfunction]
pub fn _region_read_misses() -> u64 {
    MISSES.load(std::sync::atomic::Ordering::Relaxed)
}

/// The reads of the trace running on this thread, once its body is over:
/// ``(Reads, None)``, or ``(None, why)`` when the trace depends on something
/// no check covers. ``(None, None)`` if no trace is running. ``given``: the
/// containers among the region's arguments that it checks whole, by
/// structure, before it is entered.
#[pyfunction]
#[pyo3(signature = (given=vec![]))]
pub fn _region_reads(py: Python<'_>, given: Vec<Bound<'_, PyAny>>) -> PyResult<(Option<Py<Reads>>, Option<String>)> {
    let Some(t) = TRACE.take() else { return Ok((None, None)) };
    if let Some(why) = t.refused {
        return Ok((None, Some(why)));
    }
    if let Some((d, at)) = given.iter().find_map(|x| t.filled.get(&(x.as_ptr() as usize)).map(|at| (x, at))) {
        return Ok((None, Some(format!("it asks a {} it is handed for a key it does not have, which stores the key{at}", unsafe { type_name(d.as_ptr()) }))));
    }
    let mut e = t.entries;
    unsafe {
        if let Some(c) = e.typed.iter().find(|c| ffi::Py_TYPE(c.obj.as_ptr()) as *mut PyObject != c.class.as_ptr()) {
            return Ok((None, Some(format!("the class of a {} changed while it was traced", type_name(c.obj.as_ptr())))));
        }
        if let Some(m) = e.mros.iter().find(|m| !m.holds()) {
            return Ok((None, Some(format!("the bases of {} changed while it was traced", std::ffi::CStr::from_ptr((*(m.class.as_ptr() as *mut ffi::PyTypeObject)).tp_name).to_string_lossy()))));
        }
        if e.abc.is_some_and(|was| asked::token() != was) {
            return Ok((None, Some("a class was registered with an ABC while it was traced".into())));
        }
        // fx has put back what it bound on nn.Module: what the classes have
        // now is what the body's own code saw.
        for c in &mut e.classes {
            let tp = c.class.as_ptr() as *mut ffi::PyTypeObject;
            for (name, had) in &mut c.names {
                let found = tr_type_lookup(tp, name.as_ptr());
                *had = (!found.is_null()).then(|| counted(found));
            }
        }
        // What the run read from a container it made is no entry: every
        // run of the region makes its own, and the trace's is the traced
        // call's result, which its caller may change or keep.
        e.groups.retain(|g| !super::made::is(g.holder.as_ptr()));
        // The traced run's own temporaries first: what was read from them
        // is no entry (so a getset that gives another object on every read
        // of one of them is nothing to refuse).
        sweep(&mut e);
        if let Some(r) = e.reads.iter().find(|r| r.name.is_some() && !r.holds()) {
            let name = r.name.as_ref().map_or("?".into(), |n| text(n.as_ptr()));
            return Ok((None, Some(format!("it reads .{name} of a {}, which is another object on every read{}", type_name(r.obj.as_ptr()), r.at))));
        }
        // A container the run read whole after changing it: what it holds
        // on a later call is not what the trace saw, unless the run made it
        // (swept above) or the region checks all of it as it is given (then
        // what was read is that and the run's own writes).
        if let Some(g) = e.groups.iter().find(|g| g.late && !given.iter().any(|x| x.as_ptr() == g.holder.as_ptr())) {
            let why = if ordered_ok(g.holder.as_ptr(), g.kind) { "whole after changing it" } else { "whole, and its keys are not all strs or ints" };
            return Ok((None, Some(format!("it reads a {} {why}{}", type_name(g.holder.as_ptr()), g.late_at))));
        }
        // What was no longer as it was read when the body was over (see
        // _region_settled), in something that outlives the run: a region
        // without that check would run on a state its trace did not see.
        let stale = |what: &str| -> PyResult<(Option<Py<Reads>>, Option<String>)> { Ok((None, Some(format!("it read {what} when the traced run ends, and no write the run recorded explains it")))) };
        let at = |o: &Pin| o.as_ptr() as usize;
        if let Some((_, _, what)) = t.stale.iter().find(|(kind, a, _)| match kind {
            Stale::Cell => e.reads.iter().any(|r| r.name.is_none() && at(&r.obj) == *a),
            Stale::Function => e.functions.iter().any(|f| at(&f.func) == *a),
            Stale::Dict => e.dicted.iter().any(|d| at(&d.obj) == *a),
            Stale::Group => e.groups.iter().any(|g| at(&g.holder) == *a),
        }) {
            return stale(what);
        }
        // What a cell the run made or stored into held before is no check.
        // Everything else stays as it was read, whatever is there now: what
        // the run read of a container (or an object's __dict__) it then wrote
        // to is how a later call must find it for the region's writes to be
        // the run's, and what changed since the body ended is a later call's
        // to find changed (the floats the trace puts back into the
        // arguments' containers are the region's own to check: see forget).
        e.reads.retain(|r| r.name.is_some() || r.holds() || !super::made::own_cell(r.obj.as_ptr()));
        sweep(&mut e);
        // The tensors asked about that a read found: each with the key it
        // had when it was first asked. One that no read leads to is the
        // guard's or the graph's if it is the trace's own; any other came
        // from somewhere no entry stands for, and nothing would say on a
        // later call that it is the one, or as it was.
        let found = led(&e, Start::Entries);
        for a in t.asked_of {
            // None: it died with the run.
            let o = ffi::PyWeakref_GetObject(a.weak.as_ptr());
            if o == ffi::Py_None() || (a.ours && !found.contains(&(o as usize))) {
                continue;
            }
            if !found.contains(&(o as usize)) {
                return Ok((None, Some(UNLED.into())));
            }
            let Some(key) = a.key else {
                return Ok((None, Some(format!("it asks about a {} it read from state that has no key (no plain dense tensor of exactly torch.Tensor or nn.Parameter), which it cannot check", type_name(o)))));
            };
            if !key.holds(o) {
                return stale(&format!("the class, dtype, device, shape or requires_grad of a {}, which is another", type_name(o)));
            }
            e.keyed.push(Keyed { obj: counted(o), key, ours: a.ours });
        }
        weaken(&mut e);
    }
    let reads = Reads {
        thread: t.thread,
        entries: GilCell::new(e),
        consts: GilCell::new(vec![]),
        rebind: GilCell::new(None),
        moved: Cell::new(false),
        overlap: GilCell::new(None),
        lost: Cell::new(false),
        dead: Cell::new(false),
        misses: Cell::new(0),
        last: Cell::new(""),
        hot: Cell::new(0),
    };
    Ok((Some(Py::new(py, reads)?), None))
}

// Whether the builtin `callee` reads the containers it is handed (a call
// that enters a Python frame for the callee itself hands them on unread).
pub unsafe fn reads_args(callee: *mut PyObject) -> bool {
    tr_calls_builtin(callee) != 0
}
