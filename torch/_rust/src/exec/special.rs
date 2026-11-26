// The names under which C code finds what an object's class makes of an
// operation.
//
// A builtin, or an instruction the interpreter implements in C, that is
// handed an object of a Python class asks the class: float(obj) runs what
// the class has as __float__ (or __index__), str(obj) its __str__ or
// __repr__, `obj == 1` its __eq__, sum(obj) its __iter__. Which of them a
// given piece of C code resolves, and what it falls back to, is not visible
// from outside it. So the record is by class, not by builtin: reads.rs
// notes, for the class of every such object, what the class has under each
// name of names(), and checks it when the class changed.
//
// names() is CPython's table of slots (typeobject.c's slotdefs) and the
// methods builtins look up by name. What it may not be taken for: a list of
// everything C code can read of an object. Its __dict__, its slots and the
// state of an object written in C are read without any of these names.

use std::sync::OnceLock;

use pyo3::ffi;

const NAMES: &[&std::ffi::CStr] = &[
    c"__getattribute__", c"__getattr__", c"__setattr__", c"__delattr__", c"__repr__", c"__hash__", c"__call__", c"__str__",
    c"__lt__", c"__le__", c"__eq__", c"__ne__", c"__gt__", c"__ge__", c"__iter__", c"__next__", c"__get__", c"__set__",
    c"__delete__", c"__init__", c"__new__", c"__del__", c"__buffer__", c"__release_buffer__", c"__await__", c"__aiter__",
    c"__anext__", c"__add__", c"__radd__", c"__sub__", c"__rsub__", c"__mul__", c"__rmul__", c"__mod__", c"__rmod__",
    c"__divmod__", c"__rdivmod__", c"__pow__", c"__rpow__", c"__neg__", c"__pos__", c"__abs__", c"__bool__", c"__invert__",
    c"__lshift__", c"__rlshift__", c"__rshift__", c"__rrshift__", c"__and__", c"__rand__", c"__xor__", c"__rxor__",
    c"__or__", c"__ror__", c"__int__", c"__float__", c"__iadd__", c"__isub__", c"__imul__", c"__imod__", c"__ipow__",
    c"__ilshift__", c"__irshift__", c"__iand__", c"__ixor__", c"__ior__", c"__floordiv__", c"__rfloordiv__",
    c"__truediv__", c"__rtruediv__", c"__ifloordiv__", c"__itruediv__", c"__index__", c"__matmul__", c"__rmatmul__",
    c"__imatmul__", c"__len__", c"__getitem__", c"__setitem__", c"__delitem__", c"__contains__",
    // Looked up by name.
    c"__format__", c"__round__", c"__floor__", c"__ceil__", c"__trunc__", c"__reversed__", c"__length_hint__",
    c"__bytes__", c"__complex__", c"__fspath__", c"__enter__", c"__exit__", c"__missing__", c"__copy__", c"__deepcopy__",
    c"__reduce_ex__", c"__instancecheck__", c"__subclasscheck__", c"__class__",
    // By torch's and numpy's C code.
    c"__torch_function__", c"__array__", c"__array_interface__", c"__cuda_array_interface__", c"__dlpack__",
];

// The names, interned (they live as long as the interpreter): addresses of
// str objects. No Python code runs.
pub(super) fn names() -> &'static [usize] {
    static N: OnceLock<Vec<usize>> = OnceLock::new();
    N.get_or_init(|| NAMES.iter().map(|n| unsafe { ffi::PyUnicode_InternFromString(n.as_ptr()) } as usize).filter(|n| *n != 0).collect())
}
