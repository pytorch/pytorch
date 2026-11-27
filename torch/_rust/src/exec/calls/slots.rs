// What a slot callee would run for operands of given exact classes, and
// which objects a part is read off: the Rust side of csrc/calls_slots.c,
// which says what is covered. Nothing here runs code of the program.

use std::ffi::{c_char, c_int, CStr};

use pyo3::ffi::{self, PyObject, PyTypeObject};

use super::{Callee, UNARY_OPS};

unsafe extern "C" {
    fn tr_slot_python(kind: c_int, op: c_int, a: *mut PyTypeObject, b: *mut PyTypeObject, name: *mut PyObject) -> *const c_char;
    fn tr_tuple_like(tp: *mut PyTypeObject) -> c_int;
}

// The positions, among an entry's operands, of the ones whose classes decide
// what the callee's slot runs: (the left operand, the container or the owner;
// the right operand of a binary operator or a comparison). None: no slot
// callee.
pub fn deciding(callee: &Callee) -> Option<(usize, Option<usize>)> {
    match callee {
        Callee::Binary(_) | Callee::Compare(_) => Some((0, Some(1))),
        Callee::Unary(..) | Callee::GetItem | Callee::DelItem | Callee::GetAttr(_) => Some((0, None)),
        Callee::SetItem | Callee::SetAttr(_) => Some((1, None)),
        Callee::Vector(..) | Callee::State(_) | Callee::Drop(_) => None,
    }
}

// Err: why the slot of `callee`, for operands of the classes `a` (and `b`),
// would run a method that is Python or is not covered.
pub unsafe fn resolve(callee: &Callee, a: *mut PyTypeObject, b: *mut PyTypeObject) -> Result<(), String> {
    let null = std::ptr::null_mut();
    let (kind, op, name) = match callee {
        Callee::Binary(op) => (0, *op as c_int, null),
        Callee::Compare(op) => (1, *op as c_int, null),
        Callee::Unary(opcode, oparg) => (2, UNARY_OPS.iter().position(|u| (u.1, u.2) == (*opcode, *oparg)).map_or(-1, |k| k as c_int), null),
        Callee::GetItem => (3, 0, null),
        Callee::SetItem => (4, 0, null),
        Callee::DelItem => (5, 0, null),
        Callee::GetAttr(name) => (6, 0, name.as_ptr()),
        Callee::SetAttr(name) => (7, 0, name.as_ptr()),
        Callee::Vector(..) | Callee::State(_) | Callee::Drop(_) => return Err("no slot callee".into()),
    };
    let why = unsafe { tr_slot_python(kind, op, a, b, name) };
    if why.is_null() { Ok(()) } else { Err(unsafe { CStr::from_ptr(why) }.to_string_lossy().into_owned()) }
}

// Whether the class is tuple or a struct sequence type.
pub unsafe fn tuple_class(class: *mut PyTypeObject) -> bool {
    unsafe { class == &raw mut ffi::PyTuple_Type || tr_tuple_like(class) != 0 }
}

// Whether `x` is an exact tuple or a struct sequence: what a part and a count
// are read off (a subclass of tuple can have a `__getitem__` in Python).
pub unsafe fn tuple_like(x: *mut PyObject) -> bool {
    unsafe { tuple_class(ffi::Py_TYPE(x)) }
}
