// What a question about a class reads besides its operands.
//
// isinstance, issubclass, type (of one argument), type.__instancecheck__,
// type.__subclasscheck__ and the two functions ABCMeta's methods end in
// (_abc._abc_instancecheck, _abc._abc_subclasscheck) are C code the optimizer
// takes to only compute. Their operands are found again on a later call by
// identity; the answer is also a function of
//   - which class the object is of (`obj.__class__ = Other`);
//   - the order that class (for issubclass: the first operand) resolves its
//     bases in (`cls.__bases__ = ...` makes another tuple, for the class and
//     for every class under it);
//   - what the class asked about has as __instancecheck__ or
//     __subclasscheck__: its metaclass's, whose Python code runs under the
//     trace as any frame does;
//   - for a class with ABCMeta's machinery: its registry and its caches,
//     which only C code sees. Every registration, with any ABC, moves one
//     counter (abc.get_cache_token()). A yes stays a yes while the classes
//     live: the registry only grows and the cache of yes-answers is kept. A
//     no can become a yes only after the counter moved.
//
// What it is given: the builtin a traced body called and its operands
// (reads.rs's call_read). What it answers: which question that is
// (question), which classes a second operand names (members), whether one
// has ABCMeta's machinery (abstract), the counter now (token), and whether an
// object answers `__class__` in a way of its own (answers_class_itself).
// reads.rs records the class, the order and the counter, and checks them.
//
// What it may not assume: that the object's `__class__` is its type (a
// property, a __getattribute__, a proxy written in C forward it): a question
// about such an object is refused. Not covered: abc's functions for tests
// that empty a registry or a cache (ABCMeta._abc_registry_clear,
// _abc_caches_clear) move no counter.

use std::sync::OnceLock;

use pyo3::ffi;
use pyo3::ffi::PyObject;
use pyo3::prelude::*;

use super::ffi::*;

struct Known {
    isinstance: usize,
    issubclass: usize,
    type_: usize,
    type_instancecheck: usize,
    type_subclasscheck: usize,
    abc_instancecheck: usize,
    abc_subclasscheck: usize,
    abc_meta: usize,
    get_cache_token: usize,
    union: usize,
    // What `__class__` and `__getattribute__` resolve to on a class that
    // leaves both to object, to type or to the module type.
    class_of_object: usize,
    getattribute: [usize; 3],
}

fn known() -> &'static Known {
    static K: OnceLock<Known> = OnceLock::new();
    K.get_or_init(|| {
        let py = unsafe { Python::assume_attached() };
        // The objects live as long as the interpreter: their addresses stay.
        let at = |code: &std::ffi::CStr| py.eval(code, None, None).map_or(0, |o| o.as_ptr() as usize);
        Known {
            isinstance: at(c"isinstance"),
            issubclass: at(c"issubclass"),
            type_: at(c"type"),
            type_instancecheck: at(c"type.__dict__['__instancecheck__']"),
            type_subclasscheck: at(c"type.__dict__['__subclasscheck__']"),
            abc_instancecheck: at(c"__import__('_abc')._abc_instancecheck"),
            abc_subclasscheck: at(c"__import__('_abc')._abc_subclasscheck"),
            abc_meta: at(c"__import__('abc').ABCMeta"),
            get_cache_token: at(c"__import__('_abc').get_cache_token"),
            union: at(c"__import__('types').UnionType"),
            class_of_object: at(c"object.__dict__['__class__']"),
            getattribute: [
                at(c"object.__dict__['__getattribute__']"),
                at(c"type.__dict__['__getattribute__']"),
                at(c"__import__('types').ModuleType.__dict__['__getattribute__']"),
            ],
        }
    })
}

// Finds the builtins the questions are told by. Evaluates Python code: to
// be called where no trace records (reads.rs's begin), before question.
pub(super) fn init() {
    known();
}

pub(super) struct Question {
    // The object (or, for issubclass, the class) the question is about.
    pub subject: *mut PyObject,
    // The class, tuple or union it is asked against; null for type(obj).
    pub against: *mut PyObject,
    // isinstance's kind (else issubclass's).
    pub instance: bool,
    // The call was one of ABCMeta's own.
    pub abstract_: bool,
}

// The question a call of the builtin `f` (as tr_bound_function gives it)
// with operands `all` (a bound method's object first) asks, if it is one.
pub(super) fn question(f: usize, all: &[*mut PyObject]) -> Option<Question> {
    let k = known();
    let q = |subject, against, instance, abstract_| Some(Question { subject, against, instance, abstract_ });
    match all {
        [o] if f == k.type_ => q(*o, std::ptr::null_mut(), true, false),
        [o, c] if f == k.isinstance => q(*o, *c, true, false),
        [a, c] if f == k.issubclass => q(*a, *c, false, false),
        [c, o] if f == k.type_instancecheck => q(*o, *c, true, false),
        [c, a] if f == k.type_subclasscheck => q(*a, *c, false, false),
        [c, o] if f == k.abc_instancecheck => q(*o, *c, true, true),
        [c, a] if f == k.abc_subclasscheck => q(*a, *c, false, true),
        _ => None,
    }
}

// The classes (and what else) `against` names: itself, or the members of a
// tuple or of a union, to any depth.
pub(super) unsafe fn members(against: *mut PyObject, out: &mut Vec<*mut PyObject>) {
    if against.is_null() {
    } else if ffi::PyTuple_Check(against) != 0 {
        for i in 0..ffi::PyTuple_GET_SIZE(against) {
            members(ffi::PyTuple_GET_ITEM(against, i), out);
        }
    } else if ffi::Py_TYPE(against) as usize == known().union {
        // A C getset of the union: no Python code runs. The union keeps the
        // tuple, so its members outlive this reference.
        let args = ffi::PyObject_GetAttrString(against, c"__args__".as_ptr());
        if args.is_null() {
            ffi::PyErr_Clear();
        } else {
            members(args, out);
            ffi::Py_DECREF(args);
        }
    } else {
        out.push(against);
    }
}

// Whether class `c` answers the question through ABCMeta.
pub(super) unsafe fn abstract_(c: *mut PyObject) -> bool {
    ffi::PyType_IsSubtype(ffi::Py_TYPE(c), known().abc_meta as *mut ffi::PyTypeObject) != 0
}

// Whether object `o` of class `tp` may answer `__class__` with something
// else than `tp`. A class that cannot change resolves the name in C code
// of its own (a bound method, a generic alias: the type; a weak proxy: its
// referent's): it is asked. One that can change is taken by what it has:
// a `__class__` or a `__getattribute__` of its own (asking would run its
// Python code a second time).
pub(super) unsafe fn answers_class_itself(o: *mut PyObject, tp: *mut ffi::PyTypeObject, class_name: *mut PyObject, getattribute_name: *mut PyObject) -> bool {
    if tr_type_immutable(tp) != 0 {
        let class = ffi::PyObject_GetAttr(o, class_name);
        if class.is_null() {
            ffi::PyErr_Clear();
            return true;
        }
        ffi::Py_DECREF(class);
        return class != tp as *mut PyObject;
    }
    let k = known();
    tr_type_lookup(tp, class_name) as usize != k.class_of_object || !k.getattribute.contains(&(tr_type_lookup(tp, getattribute_name) as usize))
}

// abc's counter of registrations now. C code: no Python code runs.
pub(super) unsafe fn token() -> u64 {
    let now = ffi::PyObject_CallNoArgs(known().get_cache_token as *mut PyObject);
    if now.is_null() {
        ffi::PyErr_Clear();
        return u64::MAX;
    }
    let n = ffi::PyLong_AsUnsignedLongLongMask(now);
    ffi::Py_DECREF(now);
    n
}
