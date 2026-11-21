//! Small helpers over the raw CPython API. Everything here requires the GIL.

use pyo3::ffi;
use std::os::raw::{c_char, c_int};

pub type Obj = *mut ffi::PyObject;

extern "C" {
    // Adds a traceback entry for code that has no Python frame.
    pub fn _PyTraceback_Add(funcname: *const c_char, filename: *const c_char, lineno: c_int);
    pub fn PyMethod_New(func: Obj, self_: Obj) -> Obj;
    pub static mut PyMethod_Type: ffi::PyTypeObject;
    // Borrowed references to the two halves of a bound method.
    pub fn PyMethod_Function(method: Obj) -> Obj;
    pub fn PyMethod_Self(method: Obj) -> Obj;
    pub static mut PyCell_Type: ffi::PyTypeObject;
    // New reference, or null (without an error) for an empty cell.
    pub fn PyCell_Get(cell: Obj) -> Obj;
    // value may be null, for an empty cell.
    pub fn PyCell_New(value: Obj) -> Obj;
    pub fn PyCell_Set(cell: Obj, value: Obj) -> std::os::raw::c_int;
    // New reference to sys.modules[name], or null (with or without an error).
    pub fn PyImport_GetModule(name: Obj) -> Obj;
    // The exception being handled: a new reference, or null.
    pub fn PyErr_GetHandledException() -> Obj;
    pub fn PyErr_SetHandledException(exc: Obj);
    // Borrowed reference to what the type's MRO holds for name, or null.
    pub fn _PyType_Lookup(ty: *mut ffi::PyTypeObject, name: Obj) -> Obj;
    pub fn PyType_GetDict(ty: *mut ffi::PyTypeObject) -> Obj;
    // Like getattr, but a method comes back unbound: returns 1 with the
    // function in *method, to be called with obj as its first argument.
    pub fn _PyObject_GetMethod(obj: Obj, name: Obj, method: *mut Obj) -> std::os::raw::c_int;
}

/// An owned reference to a Python object.
pub struct PyObj(pub Obj);

// PyObj is only created, cloned and dropped with the GIL held.
unsafe impl Send for PyObj {}
unsafe impl Sync for PyObj {}

impl PyObj {
    /// Takes ownership of a new reference.
    pub unsafe fn from_owned(ptr: Obj) -> PyObj {
        debug_assert!(!ptr.is_null());
        PyObj(ptr)
    }

    pub unsafe fn from_borrowed(ptr: Obj) -> PyObj {
        ffi::Py_INCREF(ptr);
        PyObj(ptr)
    }

    pub fn as_ptr(&self) -> Obj {
        self.0
    }
}

impl Clone for PyObj {
    fn clone(&self) -> PyObj {
        unsafe { PyObj::from_borrowed(self.0) }
    }
}

impl Drop for PyObj {
    fn drop(&mut self) {
        unsafe { ffi::Py_DECREF(self.0) }
    }
}

#[inline(always)]
pub unsafe fn xdecref(o: Obj) {
    if !o.is_null() {
        ffi::Py_DECREF(o);
    }
}

#[inline(always)]
pub unsafe fn new_ref(o: Obj) -> Obj {
    ffi::Py_INCREF(o);
    o
}

#[inline(always)]
pub unsafe fn py_bool(b: bool) -> Obj {
    new_ref(if b { ffi::Py_True() } else { ffi::Py_False() })
}

pub unsafe fn repr_string(o: Obj) -> String {
    if o.is_null() {
        return "<null>".to_string();
    }
    let r = ffi::PyObject_Repr(o);
    if r.is_null() {
        ffi::PyErr_Clear();
        return "<unprintable>".to_string();
    }
    let s = str_to_string(r);
    ffi::Py_DECREF(r);
    s
}

pub unsafe fn str_to_string(s: Obj) -> String {
    let mut size: ffi::Py_ssize_t = 0;
    let data = ffi::PyUnicode_AsUTF8AndSize(s, &mut size);
    if data.is_null() {
        ffi::PyErr_Clear();
        return "<?>".to_string();
    }
    let bytes = std::slice::from_raw_parts(data as *const u8, size as usize);
    String::from_utf8_lossy(bytes).into_owned()
}

extern "C" {
    pub fn _PyObject_GenericGetAttrWithDict(obj: Obj, name: Obj, dict: Obj, suppress: std::os::raw::c_int) -> Obj;
    pub fn _PyObject_GetDictPtr(obj: Obj) -> *mut Obj;
    pub fn _PyDict_MergeEx(mp: Obj, other: Obj, override_: std::os::raw::c_int) -> std::os::raw::c_int;
    pub fn _PyObject_FunctionStr(x: Obj) -> Obj;
}

/// A number that is another one once `dict` has changed.
/// Whether a dict has taken nothing but texts for keys since it was made or
/// emptied: the interpreter keeps a table for such a dict that compares no
/// key in Python code. False where that cannot be asked.
#[cfg(not(Py_3_14))]
pub unsafe fn text_keys(dict: Obj) -> bool {
    // The start of `PyDictKeysObject`: a reference count, two sizes and the
    // kind of the table, which is 0 for one that has taken another key.
    let keys = (*(dict as *mut ffi::PyDictObject)).ma_keys as *const u8;
    !keys.is_null() && *keys.add(std::mem::size_of::<isize>() + 2) != 0
}

#[cfg(Py_3_14)]
pub unsafe fn text_keys(_dict: Obj) -> bool {
    false
}

#[cfg(not(Py_3_14))]
#[inline(always)]
#[allow(deprecated)]
pub unsafe fn dict_version(dict: Obj) -> u64 {
    (*(dict as *mut ffi::PyDictObject)).ma_version_tag
}

// From 3.14 on a dict has no version of its own. The dicts that are asked
// about are watched instead, and share one version, which a change to any
// of them bumps.
#[cfg(Py_3_14)]
static WATCHED_VERSION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
/// The bit of our watcher in a dict's watcher tag; 0 until there is one.
#[cfg(Py_3_14)]
static WATCHER_BIT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg(Py_3_14)]
extern "C" {
    fn PyDict_AddWatcher(callback: unsafe extern "C" fn(c_int, Obj, Obj, Obj) -> c_int) -> c_int;
}

#[cfg(Py_3_14)]
unsafe extern "C" fn watched_dict_changed(_event: c_int, _dict: Obj, _key: Obj, _value: Obj) -> c_int {
    WATCHED_VERSION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    0
}

#[cfg(Py_3_14)]
#[inline(always)]
pub unsafe fn dict_version(dict: Obj) -> u64 {
    use std::sync::atomic::Ordering::Relaxed;
    // The start of PyDictObject; the watchers of a dict are the low bits of
    // the word that used to be its version.
    #[repr(C)]
    struct Head {
        ob_base: ffi::PyObject,
        ma_used: ffi::Py_ssize_t,
        watcher_tag: u64,
    }
    if (*(dict as *const Head)).watcher_tag & WATCHER_BIT.load(Relaxed) == 0 {
        return watch_dict(dict);
    }
    WATCHED_VERSION.load(Relaxed)
}

#[cfg(Py_3_14)]
#[cold]
unsafe fn watch_dict(dict: Obj) -> u64 {
    use std::sync::atomic::Ordering::Relaxed;
    static WATCHER: std::sync::OnceLock<c_int> = std::sync::OnceLock::new();
    let id = *WATCHER.get_or_init(|| {
        let id = PyDict_AddWatcher(watched_dict_changed);
        if id >= 0 {
            WATCHER_BIT.store(1 << id, Relaxed);
        }
        id
    });
    if id < 0 || ffi::PyDict_Watch(id, dict) < 0 {
        // The interpreter has no watcher left to give: a version that is
        // never seen twice, so nothing is assumed about the dict.
        ffi::PyErr_Clear();
        return WATCHED_VERSION.fetch_add(1, Relaxed);
    }
    WATCHED_VERSION.load(Relaxed)
}

/// How the interpreter names a callable in the errors of a call: `f()` or
/// `module.f()`.
pub unsafe fn function_str(func: Obj) -> String {
    let s = _PyObject_FunctionStr(func);
    if s.is_null() {
        ffi::PyErr_Clear();
        return "function".to_string();
    }
    let text = str_to_string(s);
    ffi::Py_DECREF(s);
    text
}

pub unsafe fn type_name(o: Obj) -> String {
    let tp = ffi::Py_TYPE(o);
    std::ffi::CStr::from_ptr((*tp).tp_name).to_string_lossy().into_owned()
}

/// `tp_name` of a class.
pub unsafe fn class_name(class: Obj) -> String {
    let tp = class as *mut ffi::PyTypeObject;
    std::ffi::CStr::from_ptr((*tp).tp_name).to_string_lossy().into_owned()
}

/// `repr(o)`; a question mark if that raises.
pub unsafe fn repr(o: Obj) -> String {
    let r = ffi::PyObject_Repr(o);
    if r.is_null() {
        ffi::PyErr_Clear();
        return "?".to_string();
    }
    let text = str_to_string(r);
    ffi::Py_DECREF(r);
    text
}

/// A cell for state that is only touched with the GIL held. Borrows must not
/// be held across calls into Python: those can re-enter the compiled function
/// or let another thread run it.
pub struct Shared<T>(std::cell::RefCell<T>);

unsafe impl<T> Send for Shared<T> {}
unsafe impl<T> Sync for Shared<T> {}

impl<T> Shared<T> {
    pub fn new(value: T) -> Shared<T> {
        Shared(std::cell::RefCell::new(value))
    }

    pub fn borrow(&self) -> std::cell::Ref<'_, T> {
        self.0.borrow()
    }

    pub fn borrow_mut(&self) -> std::cell::RefMut<'_, T> {
        self.0.borrow_mut()
    }
}
