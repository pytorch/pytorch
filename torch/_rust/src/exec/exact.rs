//! Whether two Python values are one value to every piece of code that can
//! read them: the comparison a region's checks make of what they do not
//! compare by identity (a leaf or a dict key of a container argument, a
//! pytree node's context, a quick check of an argument, a pass's guard on a
//! value, the Python side's `_same_value`).
//!
//! Given: two objects, borrowed (NULL is the same as NULL only).
//!
//! Guarantees: `same(a, b)` is true only if `a` is `b`, or both are of
//! exactly one class and that class is
//!   - int: the same number;
//!   - float: the same bits (0.0 and -0.0 are two values; a NaN is the same
//!     as a NaN of its bits);
//!   - str, bytes: the same content;
//!   - tuple, list, torch.Size: the same length and, pairwise, the same
//!     items by this rule;
//!   - torch.device: the same device type and index (a tensor's `device`
//!     is a new object on every read).
//! An object of any other class (a bool, None, a dtype, a function, an
//! instance of a subclass of the above) is the same as itself only. No
//! Python code runs.
//!
//! May not assume: that what `==` calls equal is the same (True and 1, 1
//! and 1.0, 1.0 and a numpy.float64, "a" and an instance of a str subclass,
//! an IntEnum member and its int, (1, 2) and (1, 2.0)).


use pyo3::ffi::{self, PyObject};
use pyo3::prelude::*;

// How deep containers are followed: a list can hold itself.
const DEPTH: u32 = 16;

pub unsafe fn same(a: *mut PyObject, b: *mut PyObject) -> bool {
    same_at(a, b, 0)
}

unsafe fn same_at(a: *mut PyObject, b: *mut PyObject, depth: u32) -> bool {
    if a == b {
        return true;
    }
    if a.is_null() || b.is_null() || ffi::Py_TYPE(a) != ffi::Py_TYPE(b) {
        return false;
    }
    let tp = ffi::Py_TYPE(a);
    let [size, device] = super::torch_value_types();
    if tp == &raw mut ffi::PyFloat_Type {
        ffi::PyFloat_AS_DOUBLE(a).to_bits() == ffi::PyFloat_AS_DOUBLE(b).to_bits()
    } else if tp == &raw mut ffi::PyLong_Type || tp as usize == device {
        // The class's own comparison, in C: it runs no code of the program.
        let eq = (*tp).tp_richcompare.map_or(std::ptr::null_mut(), |compare| compare(a, b, ffi::Py_EQ));
        if eq.is_null() {
            ffi::PyErr_Clear();
        }
        let same = eq == ffi::Py_True();
        ffi::Py_XDECREF(eq);
        same
    } else if tp == &raw mut ffi::PyUnicode_Type {
        ffi::PyUnicode_Compare(a, b) == 0
    } else if tp == &raw mut ffi::PyBytes_Type {
        let n = ffi::PyBytes_Size(a);
        n == ffi::PyBytes_Size(b) && std::slice::from_raw_parts(ffi::PyBytes_AsString(a), n as usize) == std::slice::from_raw_parts(ffi::PyBytes_AsString(b), n as usize)
    } else if tp == &raw mut ffi::PyTuple_Type || tp as usize == size {
        let n = ffi::PyTuple_GET_SIZE(a);
        n == ffi::PyTuple_GET_SIZE(b) && depth < DEPTH && (0..n).all(|i| same_at(ffi::PyTuple_GET_ITEM(a, i), ffi::PyTuple_GET_ITEM(b, i), depth + 1))
    } else if tp == &raw mut ffi::PyList_Type {
        let n = ffi::PyList_GET_SIZE(a);
        n == ffi::PyList_GET_SIZE(b) && depth < DEPTH && (0..n).all(|i| same_at(ffi::PyList_GET_ITEM(a, i), ffi::PyList_GET_ITEM(b, i), depth + 1))
    } else {
        false
    }
}

/// Whether ``a`` and ``b`` are one value to every reader: the same object,
/// or of exactly one class and bit for bit the same (see the module).
#[pyfunction]
pub fn _same(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> bool {
    unsafe { same(a.as_ptr(), b.as_ptr()) }
}
