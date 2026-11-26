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
//     (int index), and dict.get: the item;
//   - a cell read: the cell's content;
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
//     order of the dict's table); len() and truth: the length.
// Tuples aren't recorded: what they hold can't change.
//
// Values are compared by exact type and value if they are an int, a float
// (by bits) or a str, and by identity otherwise. The entries are grouped by
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
// Three kinds of entries are dropped rather than checked. One whose holder
// doesn't hold what was read any more when the trace ends: the traced run
// changed it itself, and it was something the run made (a list it builds,
// the object it constructs). One whose holder nothing else refers to:
// nothing can change it. And one whose holder is a list, dict or set the
// traced run made (made.rs), whoever still refers to it: a region's run
// makes its own. They keep the traced call's temporaries and results from
// being kept alive, and from failing every check.
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
use super::ffi::*;
use super::overlap::Overlap;
use super::special;
use crate::bindings::ffi as tffi;
use crate::cfg::ir::TensorSpec;

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
    value: Option<Py<PyAny>>,
}

struct Whole {
    // A dict's keys then values pairwise; a list's or a set's items.
    items: Vec<Py<PyAny>>,
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
    holder: Py<PyAny>,
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
    obj: Py<PyAny>,
    class: Py<PyAny>,
}

// A class and the tuple it resolves its bases in.
struct Mro {
    class: Py<PyAny>,
    mro: Py<PyAny>,
}

struct Class {
    class: Py<PyAny>,
    // The class's version tag when it last held.
    tag: Cell<u32>,
    // What the class has under each name looked up (None: nothing), found
    // when the trace is over.
    names: Vec<(Py<PyAny>, Option<Py<PyAny>>)>,
}

struct Read {
    // A cell (name None) or an object and the attribute a C descriptor gives.
    obj: Py<PyAny>,
    name: Option<Py<PyAny>>,
    want: Want,
    value: Option<Py<PyAny>>,
    // Where an attribute was read, for the refusal if it can't be checked.
    at: String,
}

// A Python function a frame of which ran under the traced body: the code
// that ran, and the defaults its call was filled from, are the function's
// for as long as it has them (`f.__code__ = ...`, `f.__defaults__ = ...`).
struct Function {
    func: Py<PyAny>,
    code: Py<PyAny>,
    defaults: Option<Py<PyAny>>,
    kwdefaults: Option<Py<PyAny>>,
}

// An object and the __dict__ its attributes were read from: the items read
// are the object's for as long as it has that dict (`obj.__dict__ = {...}`).
struct Dicted {
    obj: Py<PyAny>,
    dict: Py<PyAny>,
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
}

impl Entries {
    fn len(&self) -> usize {
        let groups: usize = self.groups.iter().map(|g| g.items.len() + g.whole.as_ref().map_or(0, |w| w.items.len()) + usize::from(g.len.is_some())).sum();
        let asked = self.mros.len() + usize::from(self.abc.is_some());
        groups + self.typed.len() + self.classes.iter().map(|c| c.names.len()).sum::<usize>() + self.reads.len() + self.functions.len() + self.dicted.len() + asked
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

unsafe fn equal(a: *mut PyObject, b: *mut PyObject) -> bool {
    let tp = ffi::Py_TYPE(a);
    if tp != ffi::Py_TYPE(b) {
        return false;
    }
    if tp == &raw mut ffi::PyFloat_Type {
        ffi::PyFloat_AS_DOUBLE(a).to_bits() == ffi::PyFloat_AS_DOUBLE(b).to_bits()
    } else if tp == &raw mut ffi::PyLong_Type {
        ffi::PyObject_RichCompareBool(a, b, ffi::Py_EQ) == 1
    } else if tp == &raw mut ffi::PyUnicode_Type {
        ffi::PyUnicode_Compare(a, b) == 0
    } else {
        false
    }
}

// Whether `now` (borrowed, NULL: nothing there) is what was read.
unsafe fn same(now: *mut PyObject, want: Want, value: Option<&Py<PyAny>>) -> bool {
    match want {
        Want::Absent => now.is_null(),
        Want::Present => !now.is_null(),
        Want::Is => value.is_some_and(|v| v.as_ptr() == now),
        Want::Equal => value.is_some_and(|v| v.as_ptr() == now || (!now.is_null() && equal(now, v.as_ptr()))),
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
        now.len() == self.items.len() && now.iter().zip(&self.items).all(|(a, b)| *a == b.as_ptr() || (exact_scalar(*a) && equal(*a, b.as_ptr())))
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
        self.names.iter().all(|(name, had)| tr_type_lookup(tp, name.as_ptr()) == had.as_ref().map_or(std::ptr::null_mut(), |h| h.as_ptr()))
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
        let is = |now: *mut PyObject, was: &Option<Py<PyAny>>| now == was.as_ref().map_or(std::ptr::null_mut(), |w| w.as_ptr());
        (*f).func_code == self.code.as_ptr() && is((*f).func_defaults, &self.defaults) && is((*f).func_kwdefaults, &self.kwdefaults)
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
                self.entries.groups.push(Group { holder: own(holder), kind, version: Cell::new(0), items: vec![], whole: None, len: None, late: false, late_at: String::new(), root: false });
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
        let value = (!value.is_null() && matches!(want, Want::Is | Want::Equal)).then(|| own(value));
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
        let items = items.into_iter().map(|x| own(x)).collect();
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

    unsafe fn used(&mut self, o: *mut PyObject, bound: bool) {
        if o.is_null() {
            return;
        }
        if let Some((h, kind)) = Self::container(o) {
            self.whole(h, kind);
        } else if ffi::Py_TYPE(o) == &raw mut ffi::PyDictProxy_Type {
            self.refused = Some("it reads the namespace of something that isn't a class".into());
        } else if !bound {
            self.used(tr_builtin_self(o), true);
        }
        self.handed(o);
    }

    // C code was handed `o`: what it does with an object of a Python class
    // is what the class says (see special.rs). The object's class by
    // identity, and what the class has under each name C code resolves. A
    // tensor's class is the guard's; a class that cannot change needs none.
    unsafe fn handed(&mut self, o: *mut PyObject) {
        let tp = ffi::Py_TYPE(o);
        if tr_type_immutable(tp) != 0 || tensor(o) {
            return;
        }
        if self.seen.insert((o as usize, 0)) {
            self.count += 1;
            self.entries.typed.push(Typed { obj: own(o), class: own(tp as *mut PyObject) });
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
            self.entries.typed.push(Typed { obj: own(obj), class: own(tp as *mut PyObject) });
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
                self.entries.classes.push(Class { class: own(tp as *mut PyObject), tag: Cell::new(0), names: vec![] });
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
            // Which class a tensor is of, the guard says.
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
            self.entries.mros.push(Mro { class: own(of as *mut PyObject), mro: own((*of).tp_mro) });
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
        let held = |p: *mut PyObject| (!p.is_null()).then(|| own(p));
        self.count += 1;
        self.entries.functions.push(Function { func: own(f), code: own(code), defaults: held(defaults), kwdefaults: held(kwdefaults) });
        if !kwdefaults.is_null() && ffi::PyDict_Check(kwdefaults) != 0 {
            self.whole(kwdefaults, Holder::Dict);
        }
    }

    // obj's attributes were looked up in `dict`, its __dict__.
    unsafe fn dicted(&mut self, obj: *mut PyObject, dict: *mut PyObject) {
        if !dict.is_null() && self.seen.insert((obj as usize, 2)) {
            self.count += 1;
            self.entries.dicted.push(Dicted { obj: own(obj), dict: own(dict) });
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
                        self.entries.reads.push(Read { obj: own(obj), name: Some(own(n)), want: Want::Is, value: Some(own(v)), at: place() });
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
            // A tensor's class, dtype, device, sizes, strides and
            // requires_grad are the guard's, its values the graph's. Nothing
            // else it can be asked is: an attribute the program put on the
            // object, or one of UNCHECKED_OF_A_TENSOR.
            let n = text(name);
            let mut dict = std::ptr::null_mut();
            let own_attr = tr_attr_home(obj, name, &mut dict, 1) == 2;
            ffi::Py_XDECREF(dict);
            if own_attr || UNCHECKED_OF_A_TENSOR.contains(&n.as_str()) {
                self.refused = Some(format!("it reads a tensor's .{n}"));
            } else {
                // What the class has under the name (a method the program
                // put on torch.Tensor); which class it is, the guard says.
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
                    self.entries.reads.push(Read { obj: own(obj), name: Some(own(name)), want: want_of(value), value: Some(own(value)), at: place() });
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

// container[key] was read (`ran`: by Python code, which records its own).
pub unsafe fn subscript(container: *mut PyObject, key: *mut PyObject, ran: bool) {
    with(|t| {
        if ffi::PyDict_Check(container) != 0 && key_ok(key) {
            if !ran {
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
            t.entries.reads.push(Read { obj: own(cell), name: None, want: want_of(value), value: Some(own(value)), at: String::new() });
        }
    });
}

// An instruction that ran C code was handed these: the containers among them
// are read whole.
pub unsafe fn used(objs: &[*mut PyObject]) {
    with(|t| objs.iter().for_each(|o| t.used(*o, false)));
}

// The truth (or length) of obj was taken.
pub unsafe fn truth(obj: *mut PyObject) {
    with(|t| {
        if let Some((h, kind)) = Trace::container(obj).filter(|(h, _)| *h == obj) {
            t.length(h, kind);
        } else {
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

// A tensor the region's graph reads that is none of its arguments: a weight.
// The region holds the object (the read of its home says it is still the
// one) and is entered while it has the metadata of the trace, which the
// graph is specialized on. Where its data is may change under the same
// object (`param.data = ...`): the graph reads the tensor wherever it is,
// and a lowering that was bound to the constants is bound anew first.
struct Constant {
    obj: Py<PyAny>,
    spec: TensorSpec,
    requires_grad: bool,
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
    // Checks left that also drop the entries nothing can change any more.
    sweeps: Cell<u8>,
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
        let gone = |o: &Py<PyAny>| gone.contains(&(o.as_ptr() as usize));
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
}

// Every reference the entries hold, reported to `hold`.
fn held(e: &Entries, hold: &mut impl FnMut(&Py<PyAny>)) {
    for t in &e.typed {
        hold(&t.obj);
        hold(&t.class);
    }
    for c in &e.classes {
        hold(&c.class);
        c.names.iter().for_each(|(n, had)| {
            hold(n);
            had.iter().for_each(&mut *hold);
        });
    }
    for r in &e.reads {
        hold(&r.obj);
        r.name.iter().for_each(&mut *hold);
        r.value.iter().for_each(&mut *hold);
    }
    for f in &e.functions {
        hold(&f.func);
        hold(&f.code);
        f.defaults.iter().chain(&f.kwdefaults).for_each(&mut *hold);
    }
    for d in &e.dicted {
        hold(&d.obj);
        hold(&d.dict);
    }
    for m in &e.mros {
        hold(&m.class);
        hold(&m.mro);
    }
    for g in &e.groups {
        hold(&g.holder);
        for i in &g.items {
            hold(&i.key);
            i.value.iter().for_each(&mut *hold);
        }
        g.whole.iter().flat_map(|w| &w.items).for_each(&mut *hold);
    }
}

impl Reads {
    // What kind of read the last failed check failed on.
    pub fn last(&self) -> &'static str {
        self.last.get()
    }

    // Whether everything the trace read is as it was. No Python code runs.
    pub unsafe fn check(&self) -> bool {
        let e = &mut *self.entries.get();
        if self.sweeps.get() > 0 {
            self.sweeps.set(self.sweeps.get() - 1);
            sweep(e);
        }
        let passes = |g: &Group| {
            // The order of an OrderedDict changes under the same version tag.
            let tagged = g.kind == Holder::Dict && !(g.whole.is_some() && tr_ordered(g.holder.as_ptr()) != 0);
            let v = if tagged { tr_dict_version(g.holder.as_ptr()) } else { 0 };
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
        } else if e.typed.iter().any(|t| ffi::Py_TYPE(t.obj.as_ptr()) as *mut PyObject != t.class.as_ptr()) {
            "the class of an object"
        } else if !e.classes.iter().all(|c| {
            let tp = c.class.as_ptr() as *mut ffi::PyTypeObject;
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
        } else if !e.mros.iter().all(|m| m.holds()) {
            "the bases of a class"
        } else if e.abc.is_some_and(|was| asked::token() != was) {
            "a class registered with an ABC"
        } else if let Some(i) = e.groups.iter().position(|g| !passes(g)) {
            self.hot.set(i);
            "a namespace or a container"
        } else if !e.reads.iter().all(|r| r.holds()) {
            "a cell or a slot"
        } else if !e.functions.iter().all(|f| f.holds()) {
            "the code or the defaults of a function"
        } else if e.dicted.iter().any(|d| tr_instance_dict(d.obj.as_ptr()) != d.dict.as_ptr()) {
            "the __dict__ of an object"
        } else if let Some(why) = self.constants_fail() {
            why
        } else {
            return true;
        };
        self.misses.set(self.misses.get() + 1);
        MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.last.set(why);
        false
    }

    // What a constant fails on, if one does: the metadata of the trace, or,
    // under torch.vm.Unwritten, not having been written to. For a lowering
    // bound to the constants, and for the check of how the region's tensors
    // share memory, also notes whether the data of one is somewhere else
    // than at the last look.
    unsafe fn constants_fail(&self) -> Option<&'static str> {
        let metadata = "the shape, dtype, device or requires_grad of a constant";
        let overlap = (*self.overlap.get()).as_ref();
        let watch = (*self.rebind.get()).is_some() || overlap.is_some();
        for c in &*self.consts.get() {
            let Some(t) = tffi::tensor_by_type(c.obj.as_ptr() as usize).as_ref() else { return Some(metadata) };
            if t.requires_grad() != c.requires_grad || !tffi::tensor_matches(t, c.spec.dtype, c.spec.device_type, c.spec.device_index, &c.spec.sizes, &c.spec.strides) {
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
        let i = match e.groups.iter().position(|g| g.holder.is(holder)) {
            Some(i) => i,
            None => {
                e.groups.push(Group { holder: holder.clone().unbind(), kind: Holder::Dict, version: Cell::new(0), items: vec![], whole: None, len: None, late: false, late_at: String::new(), root: false });
                e.groups.len() - 1
            }
        };
        let g = &mut e.groups[i];
        let has = g.items.iter().any(|it| it.key.is(key) || unsafe { equal(it.key.as_ptr(), key.as_ptr()) });
        let whole = g.whole.as_ref().is_some_and(|w| w.items.iter().any(|x| x.is(value)));
        if has || whole {
            return false;
        }
        g.items.push(Item { key: key.clone().unbind(), want: Want::Is, value: Some(value.clone().unbind()) });
        g.version.set(0);
        true
    }

    /// The tensors the region's graph reads that are none of its arguments
    /// (weights): the region is entered while each has the dtype, device,
    /// sizes, strides and requires_grad it has now. ``unwritten`` has, for a
    /// constant under torch.vm.Unwritten, the version counter it is to keep
    /// (None for the others). Returns how many of the constants are checked:
    /// all that are plain strided tensors.
    fn constants(&self, consts: Vec<Bound<'_, PyAny>>, unwritten: Vec<Option<i64>>) -> usize {
        let list = unsafe { &mut *self.consts.get() };
        for (x, unwritten) in consts.into_iter().zip(unwritten) {
            let Some(t) = (unsafe { tffi::tensor_by_type(x.as_ptr() as usize).as_ref() }) else { continue };
            let mut m = tffi::TensorMeta::default();
            tffi::tensor_meta(t, &mut m);
            let spec = TensorSpec { dtype: m.dtype, device_type: m.device_type, device_index: m.device_index, sizes: t.sizes().to_vec(), strides: t.strides().to_vec() };
            if tffi::tensor_matches(t, spec.dtype, spec.device_type, spec.device_index, &spec.sizes, &spec.strides) {
                list.push(Constant { obj: x.unbind(), spec, requires_grad: t.requires_grad(), at: Cell::new(tffi::tensor_data_address(t)), unwritten });
            }
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
        // Tuples aren't recorded: what one holds is read with it.
        unsafe fn reach(o: *mut PyObject, out: &mut HashSet<usize>, depth: u8) {
            out.insert(o as usize);
            if depth < 4 && ffi::PyTuple_Check(o) != 0 {
                for k in 0..ffi::PyTuple_GET_SIZE(o) {
                    reach(ffi::PyTuple_GET_ITEM(o, k), out, depth + 1);
                }
            }
        }
        let e = unsafe { &*self.entries.get() };
        let mut read: HashSet<usize> = HashSet::new();
        let mut add = |o: &Py<PyAny>| unsafe { reach(o.as_ptr(), &mut read, 0) };
        for g in &e.groups {
            g.items.iter().filter_map(|i| i.value.as_ref()).for_each(&mut add);
            g.whole.iter().flat_map(|w| &w.items).for_each(&mut add);
        }
        e.reads.iter().filter_map(|r| r.value.as_ref()).for_each(&mut add);
        e.classes.iter().flat_map(|c| &c.names).filter_map(|(_, had)| had.as_ref()).for_each(&mut add);
        (0..objects.len()).filter(|&i| read.contains(&(objects[i].as_ptr() as usize))).collect()
    }

    /// For the check of how the region's tensors share memory: from now on
    /// it is told when the data of a constant is not where it was.
    fn shares(&self, overlap: Py<Overlap>) {
        overlap.get().moved();
        unsafe { *self.overlap.get() = Some(overlap) };
    }

    /// Whether what the trace read leads to ``obj`` (see leads_to;
    /// ``given``: the region's arguments): a region that holds an object
    /// itself (a list it hands back, which was there before it ran) has the
    /// right one for as long as its reads hold.
    fn knows(&self, obj: &Bound<'_, PyAny>, given: Vec<Bound<'_, PyAny>>) -> bool {
        let given: Vec<*mut PyObject> = given.iter().map(|g| g.as_ptr()).collect();
        unsafe { leads_to(&*self.entries.get(), &given, obj.as_ptr()) }
    }

    /// Drops what was read from these containers and objects: the region
    /// checks them itself (its arguments, by structure).
    fn forget(&self, holders: Vec<Bound<'_, PyAny>>) {
        let e = unsafe { &mut *self.entries.get() };
        let given = |o: &Py<PyAny>| holders.iter().any(|h| o.is(h));
        e.groups.retain(|g| !given(&g.holder));
        e.typed.retain(|t| !given(&t.obj));
        e.reads.retain(|r| !given(&r.obj));
        e.dicted.retain(|d| !given(&d.obj));
    }

    /// The entries as text, for debugging.
    fn describe(&self, py: Python<'_>) -> Vec<String> {
        let e = unsafe { &*self.entries.get() };
        let show = |o: &Py<PyAny>| {
            let o = o.bind(py);
            let tp = o.get_type().name().map_or("?".into(), |n| n.to_string());
            if unsafe { exact_scalar(o.as_ptr()) } || o.is_none() || o.is_instance_of::<pyo3::types::PyBool>() { format!("{}", o.repr().map_or("?".into(), |r| r.to_string())) } else { format!("<{tp} at {:p}>", o.as_ptr()) }
        };
        let mut out = vec![format!("thread state {:#x}", self.thread)];
        out.extend(unsafe { &*self.consts.get() }.iter().map(|c| format!("{} keeps its shape {:?}, strides, dtype, device and requires_grad", show(&c.obj), c.spec.sizes)));
        out.extend(e.typed.iter().map(|t| format!("type({}) is {}", show(&t.obj), show(&t.class))));
        for c in &e.classes {
            let class = c.class.bind(py).getattr("__qualname__").map_or("?".into(), |n| n.to_string());
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
                Want::Absent => format!("{} not in {h}", show(&i.key)),
                Want::Present => format!("{} in {h}", show(&i.key)),
                _ => format!("{h}[{}] {} {}", show(&i.key), if i.want == Want::Is { "is" } else { "==" }, i.value.as_ref().map_or("?".into(), &show)),
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

// Whether what was read (`e`) leads to the container `p` from something
// that was there before the traced run. What was there before: the
// namespaces globals were loaded from, the cells the run read and neither
// made nor stored into, and `given` (the region's arguments). From those, by
// what the entries expect by identity: an item of a namespace or of a
// container that is led to (read alone, or with the whole content), an
// attribute of an object that is led to (an item of its __dict__, a slot,
// what its class or a base has under a name), the items of a tuple, the
// self of a bound method, the defaults of a function. That an entry was
// read FROM `p` leads nowhere: a run reads what it made too (made.rs).
unsafe fn leads_to(e: &Entries, given: &[*mut PyObject], p: *mut PyObject) -> bool {
    let mut known: HashSet<usize> = HashSet::new();
    let mut fresh: Vec<*mut PyObject> = given.to_vec();
    let is = |v: &Option<Py<PyAny>>| v.as_ref().map(|v| v.as_ptr());
    loop {
        // What an object that is led to brings with it.
        while let Some(o) = fresh.pop() {
            if o.is_null() || !known.insert(o as usize) {
                continue;
            }
            if ffi::PyTuple_Check(o) != 0 {
                fresh.extend((0..ffi::PyTuple_GET_SIZE(o)).map(|i| ffi::PyTuple_GET_ITEM(o, i)));
            } else if !tr_bound_self(o).is_null() {
                fresh.push(tr_bound_self(o));
            } else if ffi::PyFunction_Check(o) != 0 {
                fresh.push(ffi::PyFunction_GetDefaults(o));
            } else if ffi::PyList_Check(o) == 0 && ffi::PyDict_Check(o) == 0 && ffi::PyAnySet_Check(o) == 0 {
                fresh.push(instance_dict(o));
            }
            // Its class and the bases were there before it.
            let class = if ffi::PyType_Check(o) != 0 { o as *mut ffi::PyTypeObject } else { ffi::Py_TYPE(o) };
            let mro = (*class).tp_mro;
            if !mro.is_null() && known.insert(class as usize) {
                known.extend((0..ffi::PyTuple_GET_SIZE(mro)).map(|i| ffi::PyTuple_GET_ITEM(mro, i) as usize));
            }
        }
        if known.contains(&(p as usize)) {
            return true;
        }
        let led = |o: &Py<PyAny>| known.contains(&(o.as_ptr() as usize));
        for g in e.groups.iter().filter(|g| g.root || led(&g.holder)) {
            fresh.extend(g.items.iter().filter(|i| i.want == Want::Is).filter_map(|i| is(&i.value)));
            fresh.extend(g.whole.iter().flat_map(|w| w.items.iter().map(|x| x.as_ptr())));
        }
        for r in e.reads.iter().filter(|r| r.want == Want::Is) {
            let from = if r.name.is_none() { !super::made::own_cell(r.obj.as_ptr()) } else { led(&r.obj) };
            fresh.extend(is(&r.value).filter(|_| from));
        }
        for c in e.classes.iter().filter(|c| led(&c.class)) {
            fresh.extend(c.names.iter().filter_map(|(_, had)| is(had)));
        }
        fresh.retain(|o| !o.is_null() && !known.contains(&(*o as usize)));
        if fresh.is_empty() {
            return false;
        }
    }
}

/// Whether the run of the trace on this thread made the list, dict or set
/// ``obj`` (see made.rs): a region builds it anew on every call.
#[pyfunction]
pub fn _region_made(obj: &Bound<'_, PyAny>) -> bool {
    super::made::is(obj.as_ptr())
}

/// Whether the trace running on this thread came by the container ``obj``
/// through what it read, as far as it has run, from something that was
/// there before it (see leads_to; ``given``: the region's arguments): then
/// ``obj`` was there before the traced run.
#[pyfunction]
pub fn _region_known(obj: &Bound<'_, PyAny>, given: Vec<Bound<'_, PyAny>>) -> bool {
    let given: Vec<*mut PyObject> = given.iter().map(|g| g.as_ptr()).collect();
    TRACE.with_borrow(|t| t.as_ref().is_some_and(|t| unsafe { leads_to(&t.entries, &given, obj.as_ptr()) }))
}

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
                *had = (!found.is_null()).then(|| own(found));
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
        // What changed after the run read it, by other ways than the run's
        // recorded writes, is no check. What the run read of a container (or
        // an object's __dict__) it then wrote to stays as it was read: that
        // is how a later call must find it for the region's writes to be the
        // run's (the region writes to it again).
        let wrote: HashSet<usize> = t.written.iter().map(|(h, _)| *h).chain(t.changed.iter().copied()).collect();
        e.reads.retain(|r| r.holds() || r.name.as_ref().is_some_and(|n| t.written.contains(&(r.obj.as_ptr() as usize, n.as_ptr() as usize))));
        e.functions.retain(|f| f.holds());
        e.dicted.retain(|d| tr_instance_dict(d.obj.as_ptr()) == d.dict.as_ptr());
        for g in &mut e.groups {
            let (h, kind) = (g.holder.as_ptr(), g.kind);
            if wrote.contains(&(h as usize)) {
                continue;
            }
            g.items.retain(|i| i.holds(h, kind));
            if g.whole.as_ref().is_some_and(|w| !w.holds(h, kind)) {
                g.whole = None;
            }
            if g.len.is_some_and(|n| size(h, kind) != n) {
                g.len = None;
            }
        }
        sweep(&mut e);
    }
    let reads = Reads {
        thread: t.thread,
        entries: GilCell::new(e),
        consts: GilCell::new(vec![]),
        rebind: GilCell::new(None),
        moved: Cell::new(false),
        overlap: GilCell::new(None),
        sweeps: Cell::new(2),
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
