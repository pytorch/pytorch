// Side effects seen while executing a Cfg, by the site that caused them: the
// Python frames entered and the torch ops run (see torch_ops.rs). An effect is
// recorded by the innermost executor activation only, so a call to a Python
// function shows up as one frame at the caller's Call, and what the function
// does at the callee's own sites. Frames CPython runs for us (trampolines,
// generator prologues) don't record, so their effects land on the caller.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::fmt;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::Arc;

use pyo3::ffi::{self, PyObject};
use pyo3::prelude::*;

use crate::cfg::ir::{BlockId, Site, ValueId};

use super::ffi::{tr_bound_function, tr_frame_code, tr_frame_func, tr_frame_qualname, Frame, GilCell};

// Hashes ids (ValueId, Site) by multiplying, as FxHash does: they're small
// integers, which SipHash protects against nothing worth its cost here.
#[derive(Default)]
pub struct IdHasher(u64);

impl Hasher for IdHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        bytes.iter().for_each(|b| self.write_u64(*b as u64));
    }

    fn write_u64(&mut self, n: u64) {
        self.0 = (self.0.rotate_left(5) ^ n).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }

    fn write_u32(&mut self, n: u32) {
        self.write_u64(n as u64)
    }

    fn write_isize(&mut self, n: isize) {
        self.write_u64(n as u64)
    }

    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64)
    }
}

pub type IdMap<K, V> = HashMap<K, V, BuildHasherDefault<IdHasher>>;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum How {
    // The function the site called.
    Direct,
    // Called on the site's behalf by C code (a builtin, a type slot, ...).
    Indirect,
    // A generator or coroutine resumed (or thrown into).
    Resume,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Effect {
    // func is shared by every visit that entered the same function.
    Frame { func: Arc<str>, how: How },
    TorchOp { op: String, args: String },
}

impl fmt::Display for Effect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Effect::Frame { func, how: How::Direct } => write!(f, "call {func}"),
            Effect::Frame { func, how: How::Indirect } => write!(f, "call {func} (indirect)"),
            Effect::Frame { func, how: How::Resume } => write!(f, "resume {func}"),
            Effect::TorchOp { op, args } => write!(f, "torch {op}({args})"),
        }
    }
}

// A site keeps at most this many distinct sequences; past that it varies.
const MAX_SEQS: usize = 8;

#[derive(Default)]
pub struct Visits {
    pub seqs: Vec<(Vec<Effect>, u64)>,
    // Some visit produced a sequence not in `seqs`, which was full.
    pub varies: bool,
}

// For each site, the distinct sequences of effects one visit to it produced,
// with the number of visits that produced each. Visits with no effects aren't
// counted.
#[derive(Default)]
pub struct Table(IdMap<Site, Visits>);

impl Table {
    pub fn get(&self, s: Site) -> &[(Vec<Effect>, u64)] {
        self.0.get(&s).map_or(&[], |v| v.seqs.as_slice())
    }

    // Every site's sequences, and whether it varies.
    pub fn sites(&self) -> impl Iterator<Item = (&[(Vec<Effect>, u64)], bool)> {
        self.0.values().map(|v| (v.seqs.as_slice(), v.varies))
    }

    pub fn by_site(&self) -> impl Iterator<Item = (Site, &[(Vec<Effect>, u64)])> {
        self.0.iter().map(|(s, v)| (*s, v.seqs.as_slice()))
    }

    pub fn varies(&self, s: Site) -> bool {
        self.0.get(&s).is_some_and(|v| v.varies)
    }

    pub fn retain(&mut self, keep: impl Fn(Site) -> bool) {
        self.0.retain(|s, _| keep(*s));
    }

    // After the Cfg's blocks were renumbered (`map`: old id -> new id).
    pub fn renumber_blocks(&mut self, map: &[Option<BlockId>]) {
        self.0 = std::mem::take(&mut self.0)
            .into_iter()
            .filter_map(|(site, seqs)| match site {
                Site::Block(b) => map.get(b.idx()).copied().flatten().map(|n| (Site::Block(n), seqs)),
                s => Some((s, seqs)),
            })
            .collect();
    }
}

// The callable each Call site called (a bound method's function): the same
// object on every visit, or several. The optimizer identifies calls by it.
// Only callables that hold no user data are identified, held so that they
// can still be freed: functions, classes and builtins not bound to an object
// through a weakref, and method descriptors of static types (which can't be
// weakly referenced) strongly. Holding anything else, like a builtin bound to
// an object, would keep that object alive. Other calls get no entry. For a
// call of an instance whose class defines __call__ in Python, it's that
// function; for an attribute load from an instance whose class defines
// __getattr__ in Python, that function (entered only when the generic lookup
// missed: see the site's effects).
pub enum Callee {
    One { ptr: usize, weak: bool, obj: Py<PyAny>, via: Via },
    Many,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Via {
    // The callable itself.
    Callable,
    // type(callable).__call__.
    Call,
    // type(receiver).__getattr__.
    Getattr,
}

pub type Callees = IdMap<ValueId, Callee>;

impl Callee {
    // The callable, unless it's gone.
    pub fn get<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyAny>> {
        let Callee::One { weak, obj, .. } = self else {
            return None;
        };
        if !weak {
            return Some(obj.bind(py).clone());
        }
        let f = unsafe { ffi::PyWeakref_GetObject(obj.as_ptr()) };
        (f != unsafe { ffi::Py_None() }).then(|| unsafe { Bound::from_borrowed_ptr(py, f) })
    }

    pub fn describe(&self, py: Python<'_>) -> String {
        let Some(f) = self.get(py) else {
            return if matches!(self, Callee::Many) { "several".into() } else { "(gone)".into() };
        };
        let attr = |n: &str| f.getattr(n).and_then(|a| a.extract::<String>()).ok();
        match (attr("__module__"), attr("__qualname__")) {
            (Some(m), Some(q)) => format!("{m}.{q}"),
            (None, Some(q)) => q,
            _ => f.repr().map_or_else(|_| "?".into(), |r| r.to_string()),
        }
    }

    pub fn via(&self) -> Option<Via> {
        match self {
            Callee::One { via, .. } => Some(*via),
            Callee::Many => None,
        }
    }

    pub fn clone_ref(&self, py: Python<'_>) -> Callee {
        match self {
            Callee::One { ptr, weak, obj, via } => Callee::One { ptr: *ptr, weak: *weak, obj: obj.clone_ref(py), via: *via },
            Callee::Many => Callee::Many,
        }
    }
}

// `callable`: what the instr called, or (via Call or Getattr) the function
// its type runs for it.
// Whether `owner` (a builtin's self) is a pybind11 function record, which
// pybind11 binds its functions to.
unsafe fn pybind11_record(owner: *mut PyObject) -> bool {
    let name = std::ffi::CStr::from_ptr((*ffi::Py_TYPE(owner)).tp_name);
    name.to_bytes().windows(31).any(|w| w == b"pybind11_detail_function_record")
}

pub unsafe fn lru_cache_wrapper(f: *mut PyObject) -> bool {
    std::ffi::CStr::from_ptr((*ffi::Py_TYPE(f)).tp_name).to_bytes() == b"functools._lru_cache_wrapper"
}

pub unsafe fn record_callee(table: &GilCell<Callees>, v: ValueId, callable: *mut PyObject, via: Via) {
    let f = tr_bound_function(callable);
    let weak = if ffi::PyFunction_Check(f) != 0 || ffi::PyType_Check(f) != 0 || lru_cache_wrapper(f) {
        true
    } else if ffi::PyCFunction_Check(f) != 0 {
        let owner = ffi::PyCFunction_GET_SELF(f);
        // A capsule: a pybind11 function's record.
        if !(owner.is_null() || ffi::PyModule_Check(owner) != 0 || ffi::PyType_Check(owner) != 0 || ffi::PyCapsule_CheckExact(owner) != 0 || pybind11_record(owner)) {
            if std::env::var_os("TORCH_RUST_CALLEE_DEBUG").is_some() {
                let py = Python::assume_attached();
                let o = Bound::from_borrowed_ptr(py, owner);
                eprintln!("callee: builtin bound to a {}", o.get_type().name().map_or("?".into(), |n| n.to_string()));
            }
            return;
        }
        true
    } else {
        let tp = ffi::Py_TYPE(f);
        let descr = tp == &raw mut ffi::PyMethodDescr_Type
            || tp == &raw mut ffi::PyClassMethodDescr_Type
            || tp == &raw mut ffi::PyWrapperDescr_Type;
        // PyDescrObject's d_type: a static type's, or an immutable one's (a
        // C module's heap type, like itertools.chain).
        let owner = (*(f as *mut ffi::PyDescrObject)).d_type;
        if !descr || (ffi::PyType_HasFeature(owner, ffi::Py_TPFLAGS_HEAPTYPE) != 0 && ffi::PyType_HasFeature(owner, ffi::Py_TPFLAGS_IMMUTABLETYPE) == 0) {
            return;
        }
        false
    };
    let py = Python::assume_attached();
    if std::env::var_os("TORCH_RUST_CALLEE_DEBUG").is_some() {
        eprintln!("callee: %{} {}", v.0, Bound::from_borrowed_ptr(py, f).repr().map_or("?".into(), |r| r.to_string()));
    }
    let table = &mut *table.get();
    match table.get(&v) {
        Some(Callee::Many) => {}
        Some(c @ Callee::One { ptr, via: seen, .. }) if *ptr == f as usize && *seen == via && c.get(py).is_some() => {}
        Some(Callee::One { .. }) => {
            table.insert(v, Callee::Many);
        }
        None => {
            let obj = if weak { ffi::PyWeakref_NewRef(f, std::ptr::null_mut()) } else { ffi::Py_NewRef(f) };
            if obj.is_null() {
                ffi::PyErr_Clear();
                return;
            }
            table.insert(v, Callee::One { ptr: f as usize, weak, obj: Py::from_owned_ptr(py, obj), via });
        }
    }
}

// One executor activation's view: the site it is at and the effects of the
// current visit to it.
pub struct Recorder {
    table: *const GilCell<Table>,
    site: Site,
    pending: Vec<Effect>,
    // The callable of the Call being run, until the frame it enters directly.
    callee: *mut PyObject,
    prev: *mut Recorder,
}

thread_local! {
    static CURRENT: Cell<*mut Recorder> = const { Cell::new(std::ptr::null_mut()) };
}

impl Recorder {
    // At every instr, so the common case (nothing pending) is inline.
    #[inline]
    fn commit(&mut self) {
        if !self.pending.is_empty() {
            self.commit_pending();
        }
    }

    #[inline(never)]
    fn commit_pending(&mut self) {
        // With the GIL held (pending only collects without it).
        let table = unsafe { &mut *(*self.table).get() };
        let visits = table.0.entry(self.site).or_default();
        if let Some((_, n)) = visits.seqs.iter_mut().find(|(s, _)| *s == self.pending) {
            *n += 1;
        } else if visits.seqs.len() < MAX_SEQS {
            visits.seqs.push((self.pending.clone(), 1));
        } else {
            visits.varies = true;
        }
        self.pending.clear();
    }

    // Before the recorder is set aside (see run.rs's run_blocks).
    pub fn flush(&mut self) {
        self.commit();
    }

    #[inline]
    pub fn at(&mut self, site: Site) {
        if self.site != site {
            self.commit();
            self.site = site;
        }
    }

    pub fn set_callee(&mut self, callee: *mut PyObject) {
        self.callee = callee;
    }
}

struct Scope(*mut Recorder);

impl Drop for Scope {
    fn drop(&mut self) {
        unsafe {
            (*self.0).commit();
            set_current((*self.0).prev);
        }
    }
}

// Makes `rec` (null: none) the innermost recorder on this thread; returns the
// previous one. The op hook only runs while there is one, since it costs a
// boxed redispatch per op.
pub fn set_current(rec: *mut Recorder) -> *mut Recorder {
    let old = CURRENT.replace(rec);
    if old.is_null() != rec.is_null() && !super::torch_ops::printing() {
        crate::bindings::ffi::op_hook::op_hook_pause(rec.is_null());
    }
    old
}

// Runs `f` with a recorder for `table`, starting at `site`, as the innermost
// one on this thread.
pub fn scope<R>(table: &GilCell<Table>, site: Site, f: impl FnOnce(*mut Recorder) -> R) -> R {
    let mut rec = Recorder {
        table,
        site,
        pending: vec![],
        callee: std::ptr::null_mut(),
        prev: CURRENT.get(),
    };
    let p: *mut Recorder = &mut rec;
    set_current(p);
    let _scope = Scope(p);
    f(p)
}

pub fn has_current() -> bool {
    !CURRENT.get().is_null()
}

// Runs `f` with no recorder on this thread.
pub fn unrecorded<R>(f: impl FnOnce() -> R) -> R {
    struct Restore(*mut Recorder);
    impl Drop for Restore {
        fn drop(&mut self) {
            set_current(self.0);
        }
    }
    let _restore = Restore(set_current(std::ptr::null_mut()));
    f()
}

// Before the eval-frame hook runs `frame`, holding the GIL.
pub unsafe fn frame_entered(frame: *mut Frame, resumed: bool) {
    let rec = CURRENT.get();
    if rec.is_null() {
        return;
    }
    let func = tr_frame_func(frame);
    let how = if resumed {
        How::Resume
    } else if !(*rec).callee.is_null() && tr_bound_function((*rec).callee) == func {
        (*rec).callee = std::ptr::null_mut();
        How::Direct
    } else {
        How::Indirect
    };
    let func = func_name(func, tr_frame_qualname(frame));
    note_code(&func, tr_frame_code(frame));
    (*rec).pending.push(Effect::Frame { func, how });
}

// The code of each function a frame was entered for, by name (0 for a name
// several codes had), for judging whether a call is pure when its callee
// isn't known (entered indirectly, resumed, or a closure that's gone). The
// codes are kept alive.
static CODES: std::sync::Mutex<Option<HashMap<Arc<str>, usize>>> = std::sync::Mutex::new(None);

unsafe fn note_code(func: &Arc<str>, code: *mut PyObject) {
    let mut guard = CODES.lock().unwrap();
    let map = guard.get_or_insert_with(HashMap::new);
    match map.get(func) {
        Some(&c) if c == code as usize || c == 0 => {}
        Some(_) => {
            map.insert(func.clone(), 0);
        }
        None => {
            ffi::Py_INCREF(code);
            map.insert(func.clone(), code as usize);
        }
    }
}

// The code of frames entered as `func`, if only one was.
pub fn code_named<'py>(py: Python<'py>, func: &str) -> Option<Bound<'py, PyAny>> {
    let guard = CODES.lock().unwrap();
    let &c = guard.as_ref()?.get(func)?;
    (c != 0).then(|| unsafe { Bound::from_borrowed_ptr(py, c as *mut PyObject) })
}

// "module.qualname", read from the function and code objects (a frame's
// function is always a PyFunction, whose module may be NULL), cached by code
// object. An entry holds the module and qualname objects it was made from,
// so while they're the frame's the name is the same.
unsafe fn func_name(func: *mut PyObject, qualname: *mut PyObject) -> Arc<str> {
    struct Entry {
        module: *mut PyObject,
        qualname: *mut PyObject,
        name: Arc<str>,
    }
    thread_local! {
        // Direct-mapped by qualname address. The objects it holds are never
        // released: a thread's few hundred strs at most.
        static CACHE: RefCell<Vec<Option<Entry>>> = RefCell::new((0..512).map(|_| None).collect());
    }
    let module = if !func.is_null() && ffi::PyFunction_Check(func) != 0 { ffi::PyFunction_GetModule(func) } else { std::ptr::null_mut() };
    let slot = (qualname as usize >> 4) % 512;
    let hit = CACHE.with_borrow(|c| match &c[slot] {
        Some(e) if e.module == module && e.qualname == qualname => Some(e.name.clone()),
        _ => None,
    });
    if let Some(name) = hit {
        return name;
    }
    let name: Arc<str> = make_func_name(module, qualname).into();
    // Only strs: a module object of another type could be user data.
    if module.is_null() || ffi::PyUnicode_CheckExact(module) != 0 {
        let old = CACHE.with_borrow_mut(|c| {
            ffi::Py_XINCREF(module);
            ffi::Py_INCREF(qualname);
            c[slot].replace(Entry { module, qualname, name: name.clone() })
        });
        if let Some(old) = old {
            ffi::Py_XDECREF(old.module);
            ffi::Py_DECREF(old.qualname);
        }
    }
    name
}

// A throw into a generator enters with the exception set, so it's put aside
// around the conversions.
unsafe fn make_func_name(module: *mut PyObject, qualname: *mut PyObject) -> String {
    let saved = ffi::PyErr_GetRaisedException();
    let py = Python::assume_attached();
    let qualname = Bound::from_borrowed_ptr(py, qualname).extract::<String>().unwrap_or_else(|_| "?".into());
    let module = Bound::from_borrowed_ptr_or_opt(py, module).and_then(|m| m.extract::<String>().ok());
    ffi::PyErr_SetRaisedException(saved);
    match module {
        Some(m) => format!("{m}.{qualname}"),
        None => qualname,
    }
}

// From the PreDispatch hook, maybe without the GIL.
pub fn torch_op(op: &str, args: &str) {
    let rec = CURRENT.get();
    if !rec.is_null() {
        let op = Effect::TorchOp {
            op: op.into(),
            args: args.into(),
        };
        unsafe { (*rec).pending.push(op) };
    }
}
