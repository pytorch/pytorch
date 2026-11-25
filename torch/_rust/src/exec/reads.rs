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
//   - any other instruction or builtin handed a dict, a list, a set or a dict
//     view (iterating it, unpacking it, slicing it, zip, sorted, ...): the
//     whole content, in order; len() and truth: the length.
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
// Two kinds of entries are dropped rather than checked. One whose holder
// doesn't hold what was read any more when the trace ends: the traced run
// changed it itself, and a region's run only writes to what it made (a list
// it builds, the object it constructs). And one whose holder nothing else
// refers to: nothing can change it. Both keep the traced call's temporaries
// from being kept alive, and from failing every check.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};

use pyo3::ffi;
use pyo3::ffi::PyObject;
use pyo3::prelude::*;

use super::ffi::*;
use crate::bindings::ffi as tffi;

// Entries a region may have, and items a container read whole may have.
const MAX_ENTRIES: usize = 1 << 15;
const MAX_ITEMS: usize = 1 << 12;

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
}

struct Typed {
    obj: Py<PyAny>,
    class: Py<PyAny>,
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

#[derive(Default)]
struct Entries {
    groups: Vec<Group>,
    typed: Vec<Typed>,
    classes: Vec<Class>,
    reads: Vec<Read>,
}

impl Entries {
    fn len(&self) -> usize {
        let groups: usize = self.groups.iter().map(|g| g.items.len() + g.whole.as_ref().map_or(0, |w| w.items.len()) + usize::from(g.len.is_some())).sum();
        groups + self.typed.len() + self.classes.iter().map(|c| c.names.len()).sum::<usize>() + self.reads.len()
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
    TRACE.set(Some(Trace { thread: tffi::region_thread_state(), ..Trace::default() }));
}

pub fn end() {
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

unsafe fn own(o: *mut PyObject) -> Py<PyAny> {
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

unsafe fn tensor(o: *mut PyObject) -> bool {
    !tffi::tensor_by_type(o as usize).is_null()
}

unsafe fn text(o: *mut PyObject) -> String {
    let mut n: ffi::Py_ssize_t = 0;
    let p = if ffi::PyUnicode_Check(o) != 0 { ffi::PyUnicode_AsUTF8AndSize(o, &mut n) } else { std::ptr::null() };
    if p.is_null() {
        ffi::PyErr_Clear();
        return "?".into();
    }
    String::from_utf8_lossy(std::slice::from_raw_parts(p as *const u8, n as usize)).into_owned()
}

unsafe fn type_name(o: *mut PyObject) -> String {
    std::ffi::CStr::from_ptr((*ffi::Py_TYPE(o)).tp_name).to_string_lossy().into_owned()
}

// The item of dict d at key k (borrowed), NULL if absent; no Python code runs
// for a str or int key.
unsafe fn dict_get(d: *mut PyObject, k: *mut PyObject) -> *mut PyObject {
    let x = ffi::PyDict_GetItemWithError(d, k);
    if x.is_null() && !ffi::PyErr_Occurred().is_null() {
        ffi::PyErr_Clear();
    }
    x
}

unsafe fn snapshot(holder: *mut PyObject, kind: Holder) -> Vec<*mut PyObject> {
    let mut out = vec![];
    match kind {
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
        self.items.is_empty() && self.whole.is_none() && self.len.is_none()
    }
}

impl Class {
    // Whether the class still has what it had under each name.
    unsafe fn holds(&self) -> bool {
        let tp = self.class.as_ptr() as *mut ffi::PyTypeObject;
        self.names.iter().all(|(name, had)| tr_type_lookup(tp, name.as_ptr()) == had.as_ref().map_or(std::ptr::null_mut(), |h| h.as_ptr()))
    }
}

impl Read {
    unsafe fn holds(&self) -> bool {
        match &self.name {
            None => same(PyCell_Get_borrowed(self.obj.as_ptr()), self.want, self.value.as_ref()),
            Some(name) => {
                let now = ffi::PyObject_GetAttr(self.obj.as_ptr(), name.as_ptr());
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
                self.entries.groups.push(Group { holder: own(holder), kind, version: Cell::new(0), items: vec![], whole: None, len: None });
                self.group_of.insert(holder as usize, self.entries.groups.len() - 1);
                self.entries.groups.len() - 1
            }
        };
        &mut self.entries.groups[i]
    }

    // The first read of an item is the one recorded: a later one that differs
    // is of something the traced run changed.
    unsafe fn item(&mut self, holder: *mut PyObject, kind: Holder, key: *mut PyObject, want: Want, value: *mut PyObject) {
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
        if self.group_of.get(&(holder as usize)).is_some_and(|i| self.entries.groups[*i].whole.is_some()) {
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

    // An attribute lookup on an instance of tp: which code resolves it.
    unsafe fn resolves(&mut self, obj: *mut PyObject) {
        self.special(obj, c"__getattribute__");
        self.special(obj, c"__getattr__");
    }

    unsafe fn attr(&mut self, obj: *mut PyObject, name: *mut PyObject, value: *mut PyObject, ran: bool) {
        if tensor(obj) {
            // A tensor's dtype, device, sizes and requires_grad are the
            // guard's or the graph's. What autograd keeps on it is not.
            let n = text(name);
            if matches!(n.as_str(), "grad" | "grad_fn" | "_grad" | "_base") {
                self.refused = Some(format!("it reads a tensor's .{n}"));
            }
            return;
        }
        let mut dict = std::ptr::null_mut();
        let home = tr_attr_home(obj, name, &mut dict);
        match home {
            0 => {}
            // A method, a class attribute, the class's __getattr__: the class,
            // and the instance not having it itself.
            1 => {
                self.typed(obj, name);
                self.resolves(obj);
                if !dict.is_null() {
                    self.item(dict, Holder::Dict, name, Want::Absent, std::ptr::null_mut());
                }
            }
            2 => {
                self.typed(obj, name);
                self.resolves(obj);
                self.dict_item(dict, name, false);
            }
            3 => {
                self.typed(obj, name);
                self.resolves(obj);
                let n = text(name);
                if !ran && !value.is_null() && n != "__dict__" && n != "__class__" && self.seen.insert((obj as usize, name as usize)) {
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
                    self.refused = Some(format!("it reads .{} of a {}, which it cannot check", text(name), type_name(obj)));
                }
            }
        }
        ffi::Py_XDECREF(dict);
    }
}

// The builtins whose reads are recorded by item rather than whole.
struct Builtins {
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
        } else {
            t.item(globals, Holder::Dict, name, want_of(value), value);
        }
    });
}

// obj.name was loaded: `value` (NULL: it isn't there), `ran` if that entered
// a Python frame (a property, __getattr__), whose own reads are recorded.
// The exception set, if any, is kept.
pub unsafe fn attr(obj: *mut PyObject, name: *mut PyObject, value: *mut PyObject, ran: bool) {
    let raised = if value.is_null() { ffi::PyErr_GetRaisedException() } else { std::ptr::null_mut() };
    with(|t| t.attr(obj, name, value, ran));
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

// obj, which is no function, was called: its class's __call__ ran.
pub unsafe fn called(obj: *mut PyObject) {
    with(|t| t.special(obj, c"__call__"));
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
// and returned `result`.
pub unsafe fn call(callee: *mut PyObject, args: &[*mut PyObject], result: *mut PyObject, ran: bool) {
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
        with(|t| t.attr(all[0], all[1], value, ran));
    } else if (f == b.dict_get || f == b.dict_getitem || f == b.dict_contains) && all.len() >= 2 && ffi::PyDict_Check(all[0]) != 0 && key_ok(all[1]) {
        with(|t| t.dict_item(all[0], all[1], f == b.dict_contains));
    } else if f == b.len && all.len() == 1 {
        truth(all[0]);
    } else {
        used(&all);
    }
}

// What a region is checked against when it is entered (see the top of the
// file). Made by torch._rust._region_reads when the body's trace is over.
#[pyclass(frozen, module = "torch._rust")]
pub struct Reads {
    thread: u64,
    entries: GilCell<Entries>,
    // Checks left that also drop the entries nothing can change any more.
    sweeps: Cell<u8>,
    misses: Cell<u64>,
    last: Cell<&'static str>,
    // The group the last failed check failed on, looked at first: a region
    // traced for a state that is gone fails there again.
    hot: Cell<usize>,
}

unsafe impl Sync for Reads {}

unsafe fn sweep(e: &mut Entries) {
    let alone = |o: &Py<PyAny>| ffi::Py_REFCNT(o.as_ptr()) == 1;
    // An object first: its __dict__ is only its own once it is gone.
    e.typed.retain(|t| !alone(&t.obj));
    e.reads.retain(|r| !alone(&r.obj));
    e.groups.retain(|g| !alone(&g.holder) && !g.empty());
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
            let v = if g.kind == Holder::Dict { tr_dict_version(g.holder.as_ptr()) } else { 0 };
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
        } else if let Some(i) = e.groups.iter().position(|g| !passes(g)) {
            self.hot.set(i);
            "a namespace or a container"
        } else if !e.reads.iter().all(|r| r.holds()) {
            "a cell or a slot"
        } else {
            return true;
        };
        self.misses.set(self.misses.get() + 1);
        MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.last.set(why);
        false
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
                e.groups.push(Group { holder: holder.clone().unbind(), kind: Holder::Dict, version: Cell::new(0), items: vec![], whole: None, len: None });
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

    /// Drops what was read from these containers and objects: the region
    /// checks them itself (its arguments, by structure).
    fn forget(&self, holders: Vec<Bound<'_, PyAny>>) {
        let e = unsafe { &mut *self.entries.get() };
        let given = |o: &Py<PyAny>| holders.iter().any(|h| o.is(h));
        e.groups.retain(|g| !given(&g.holder));
        e.typed.retain(|t| !given(&t.obj));
        e.reads.retain(|r| !given(&r.obj));
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
        out.extend(e.reads.iter().map(|r| match &r.name {
            None => format!("{} holds {}", show(&r.obj), r.value.as_ref().map_or("?".into(), &show)),
            Some(n) => format!("{}.{} {} {}{}", show(&r.obj), n.bind(py), if r.want == Want::Is { "is" } else { "==" }, r.value.as_ref().map_or("?".into(), &show), r.at),
        }));
        out
    }
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
/// no check covers. ``(None, None)`` if no trace is running.
#[pyfunction]
pub fn _region_reads(py: Python<'_>) -> PyResult<(Option<Py<Reads>>, Option<String>)> {
    let Some(t) = TRACE.take() else { return Ok((None, None)) };
    if let Some(why) = t.refused {
        return Ok((None, Some(why)));
    }
    let mut e = t.entries;
    unsafe {
        if let Some(c) = e.typed.iter().find(|c| ffi::Py_TYPE(c.obj.as_ptr()) as *mut PyObject != c.class.as_ptr()) {
            return Ok((None, Some(format!("the class of a {} changed while it was traced", type_name(c.obj.as_ptr())))));
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
        if let Some(r) = e.reads.iter().find(|r| r.name.is_some() && !r.holds()) {
            let name = r.name.as_ref().map_or("?".into(), |n| text(n.as_ptr()));
            return Ok((None, Some(format!("it reads .{name} of a {}, which is another object on every read{}", type_name(r.obj.as_ptr()), r.at))));
        }
        // What the traced run changed after reading it was its own.
        e.reads.retain(|r| r.holds());
        for g in &mut e.groups {
            let (h, kind) = (g.holder.as_ptr(), g.kind);
            g.items.retain(|i| i.holds(h, kind));
            if g.whole.as_ref().is_some_and(|w| !w.holds(h, kind)) {
                g.whole = None;
            }
            if g.len.is_some_and(|n| size(h, kind) != n) {
                g.len = None;
            }
        }
        sweep(&mut e);
        sweep(&mut e);
    }
    let reads = Reads { thread: t.thread, entries: GilCell::new(e), sweeps: Cell::new(2), misses: Cell::new(0), last: Cell::new(""), hot: Cell::new(0) };
    Ok((Some(Py::new(py, reads)?), None))
}

// Whether the builtin `callee` reads the containers it is handed (a call
// that enters a Python frame for the callee itself hands them on unread).
pub unsafe fn reads_args(callee: *mut PyObject) -> bool {
    tr_calls_builtin(callee) != 0
}
