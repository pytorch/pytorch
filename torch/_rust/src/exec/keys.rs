//! Whether asking a dict or a set about a key runs code of the program.
//!
//! The recorder of a trace's writes (writes.rs) asks a container what it
//! holds under a key before the traced code stores or deletes there: what
//! it finds is what an undo puts back. A lookup hashes the key and compares
//! it with what the container already has under that hash. Both are code
//! of the key's class where the class defines them in Python, and such
//! code is the program's: run by the recorder it would run once more than
//! the program runs it, and inside the recorder's own records.
//!
//! Given: an object, a dict (of any class) or a set (of any class), borrowed.
//!
//! Guarantees: `plain(key)` is true only if hashing `key`, and comparing it
//! with any other plain object, is C code of a builtin class that reads the
//! two objects alone: `key` is None, or its class has for both the hash and
//! the comparison the C function of str, int, float, bytes or `object`
//! (those classes, and a subclass that defines neither `__eq__` nor
//! `__hash__`: an instance of a plain class goes by its address), or those
//! of tuple and every item is plain (to DEPTH).
//! `holds_plain(holder)` is true only if every key of the dict, or every
//! member of the set, is plain. A lookup of a plain key in a holder that
//! holds plain keys runs no Python code. No Python code runs here.
//!
//! May not assume: that an instance of a subclass of str or int hashes as
//! its base does (a subclass may define `__hash__` and `__eq__`: the class's
//! slots are asked, which CPython keeps in step with what the class
//! defines); that a plain key meets only plain keys (hence `holds_plain`);
//! that a holder that holds plain keys now does so on a later call.

use pyo3::ffi::{self, PyObject};

use super::ffi::_PySet_NextEntry;

// How deep tuples are followed.
const DEPTH: u32 = 8;

pub unsafe fn plain(key: *mut PyObject) -> bool {
    plain_at(key, 0)
}

unsafe fn plain_at(key: *mut PyObject, depth: u32) -> bool {
    if key == ffi::Py_None() {
        return true;
    }
    let tp = ffi::Py_TYPE(key);
    let slots = |tp: *mut ffi::PyTypeObject| ((*tp).tp_hash.map(|f| f as usize), (*tp).tp_richcompare.map(|f| f as usize));
    let mine = slots(tp);
    let scalars = [&raw mut ffi::PyUnicode_Type, &raw mut ffi::PyLong_Type, &raw mut ffi::PyFloat_Type, &raw mut ffi::PyBytes_Type, &raw mut ffi::PyBaseObject_Type];
    if scalars.iter().any(|base| slots(*base) == mine) {
        return true;
    }
    slots(&raw mut ffi::PyTuple_Type) == mine
        && ffi::PyTuple_Check(key) != 0
        && depth < DEPTH
        && (0..ffi::PyTuple_GET_SIZE(key)).all(|i| plain_at(ffi::PyTuple_GET_ITEM(key, i), depth + 1))
}

pub unsafe fn holds_plain(holder: *mut PyObject) -> bool {
    let (mut pos, mut key, mut value, mut hash) = (0, std::ptr::null_mut(), std::ptr::null_mut(), 0);
    if ffi::PyDict_Check(holder) != 0 {
        while ffi::PyDict_Next(holder, &mut pos, &mut key, &mut value) != 0 {
            if !plain(key) {
                return false;
            }
        }
        return true;
    }
    if ffi::PyAnySet_Check(holder) == 0 {
        return false;
    }
    while _PySet_NextEntry(holder, &mut pos, &mut key, &mut hash) != 0 {
        if !plain(key) {
            return false;
        }
    }
    true
}
