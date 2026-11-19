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
