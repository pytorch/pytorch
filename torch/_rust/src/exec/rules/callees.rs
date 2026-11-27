// The callees no family has a line for: the kind of callable each is.
//
// A call is asked about as its callee (ops::what). A family answers for the
// callees it lists by identity (torch.rs: the C callees of torch that were
// measured). Every other callee comes here, and is told apart by the exact
// type of the callee object, following what CPython itself does with a call
// of that type: enter a function's frame, call a C function, make an
// instance of a class, look `__call__` up on the type. Each kind is left, by
// name, to the family of rules that will take calls of it; until that family
// is written the code that decided before the table decides. A callable of
// none of the kinds written down here has a kind of its own, the last of the
// list, and that kind is a STOP: no rule says what a call of it runs, and
// none is to be written.
//
// What it is given
//   - by rules::said, a callee (never NULL) that no family answered for;
//   - by ready(), called where a function goes onto the executor and never
//     under a trace: the moment to find the C callables of torch's own
//     namespaces (attribute loads, in Python, once).
// What it guarantees
//   - of() compares the address of the callee's type, reads a flag of that
//     type, and for an instance looks `__call__` up on its type in C
//     (_PyType_Lookup): no code of the program runs, and nothing is made on
//     first use;
//   - "of torch's namespaces" is identity: the C definition a builtin or a
//     descriptor is made from is one that ready() found there. It is a NAME
//     for a report and for the day those callees stop a trace; it licenses
//     nothing;
//   - before ready() ran, a C function of torch is told as any builtin and
//     a partial, a bound slot wrapper or an instance as "anything else",
//     which stops: nothing is looked up lazily. No trace runs then
//     (rules::ready is called where a function goes onto the executor), and
//     the doors for tests call it first;
//   - of() is Later for every kind but the last, which is a Stop.
// What it may not assume
//   - that a kind says what the call does: a builtin may call back into
//     Python, a class may have any `__new__`. The kind only names who is to
//     say.

use std::collections::HashSet;
use std::ffi::CStr;
use std::sync::OnceLock;

use pyo3::ffi;
use pyo3::ffi::PyObject;
use pyo3::prelude::*;

use super::super::ffi::{tr_partial_func, tr_type_lookup, PyMethod_Type};
use super::Taken;

// The kinds, in the order of() tries them: what the dump calls the kind,
// what a call of it is left to (or that it stops), and why.
pub const KINDS: &[(&str, Taken, &str)] = &[
    ("function", Taken::Later("callees: python functions"), "A call of a Python function enters its frame, and the instructions of that frame are asked about one by one; how the arguments are bound to it is the rule to come."),
    ("method", Taken::Later("callees: bound methods"), "A call of a bound method is a call of what it is bound from with its receiver in front: the kind of that callable decides."),
    ("a C function or descriptor of torch's namespaces", Taken::Later("callees: torch, unswept"), "A C callable of torch that no sweep measured: nothing says what it writes, draws or follows."),
    ("builtin_function_or_method", Taken::Later("callees: builtins"), "A C function, alone or bound to its receiver: what it reads of what it is handed, and whether it calls back, is a fact of each one, to be written by identity."),
    ("method_descriptor", Taken::Later("callees: method descriptors"), "A C method of a class, called with its receiver as the first operand: as a builtin, a fact of each one."),
    ("wrapper_descriptor", Taken::Later("callees: slot wrappers"), "A slot of a C class as a callable, alone or bound to its receiver (`int.__add__`, `x.__eq__`): as a builtin, a fact of each one."),
    ("type", Taken::Later("callees: classes"), "A call of a class whose metaclass is exactly type makes an instance: what the class has under __new__ and __init__ decides which code runs."),
    ("a class of another metaclass", Taken::Later("callees: classes of a metaclass"), "A call of a class is what its metaclass has under __call__: for a metaclass other than type that may be any code (an Enum looks a member up)."),
    ("functools.partial", Taken::Later("callees: partials"), "A call of an exact functools.partial is a call of what it holds, with the arguments it holds in front: the kind of that callable decides."),
    ("an instance of a class with a Python __call__", Taken::Later("callees: instances"), "The class of the object has a Python function under __call__ (a module, an object with __call__): a call enters that function's frame."),
    ("an instance of a class with a C __call__", Taken::Later("callees: C callables"), "The class of the object has a slot of C code under __call__ (a cached function, a staticmethod object, an operator.itemgetter): as a builtin, a fact of each class."),
    ("anything else", Taken::Stop("it is a call of an object whose class has under __call__ neither a Python function nor a slot of C code, or nothing at all: no rule says what that call runs"), "An object whose class has under __call__ neither a Python function nor a slot of C code (a builtin, a partial, a staticmethod, a classmethod object), or nothing at all: no kind above says what a call of it runs, and no rule is to be written for it."),
];

struct Made {
    // What identifies the C functions of torch's namespaces (see key).
    torch: HashSet<usize>,
    // `__call__`, interned.
    call: usize,
    // The type of a slot wrapper bound to an object (`(1).__add__`).
    bound_wrapper: usize,
}

static MADE: OnceLock<Made> = OnceLock::new();

// The C callables that torch's own namespaces hold, by the sweep's list of
// namespaces: `found` is a list of them. A C function that a class of torch
// inherits from object or type, or that a module of torch has from another
// module, is not torch's (every object's `__dir__` is one C function): the
// class a descriptor is of, the module a function is of, or the class of
// the object a method is bound to that defines the name, says whose it is.
const NAMESPACES: &CStr = c"
bound_wrapper = type(object().__init__)
import types, torch, torch.nn.functional
C = ('builtin_function_or_method', 'method_descriptor', 'wrapper_descriptor', 'classmethod_descriptor')
spaces = (torch, torch.Tensor, torch._C._nn, torch._C._linalg, torch._C._fft, torch._C._special, torch._C._sparse,
          torch._C._VariableFunctions, torch._C, torch.nn.functional, torch.linalg, torch.fft, torch.special, torch.sparse)
def of_torch(name, member):
    owner = getattr(member, '__objclass__', None)
    if owner is None:
        bound = getattr(member, '__self__', None)
        if bound is None:
            return True
        if isinstance(bound, types.ModuleType):
            return bound.__name__.startswith('torch')
        cls = bound if isinstance(bound, type) else type(bound)
        owner = next((k for k in cls.__mro__ if name in vars(k)), None)
        if owner is None:
            return True
    return (owner.__module__ or '').startswith('torch')
found = []
for space in spaces:
    for name in dir(space):
        try:
            member = getattr(space, name)
            if type(member).__name__ in C and of_torch(name, member):
                found.append(member)
        except Exception:
            continue
";

// What identifies the C function behind a callable: for a builtin and a
// descriptor the definition they are made from, else the object.
unsafe fn key(f: *mut PyObject) -> usize {
    let tp = ffi::Py_TYPE(f);
    if tp == &raw mut ffi::PyCFunction_Type {
        (*(f as *mut ffi::PyCFunctionObject)).m_ml as usize
    } else if tp == &raw mut ffi::PyMethodDescr_Type || tp == &raw mut ffi::PyClassMethodDescr_Type {
        (*(f as *mut ffi::PyMethodDescrObject)).d_method as usize
    } else {
        f as usize
    }
}

// Finds the C callables of torch's namespaces and the type of partials.
// Runs Python code: called where a function goes onto the executor, not
// from inside it (rules::ready).
pub fn ready() {
    if MADE.get().is_some() {
        return;
    }
    let made = Python::attach(|py| unsafe {
        let g = pyo3::types::PyDict::new(py);
        let ran = py.run(NAMESPACES, Some(&g), None);
        let found = super::super::reads::dict_get(g.as_ptr(), ffi::PyUnicode_InternFromString(c"found".as_ptr()));
        let bound_wrapper = super::super::reads::dict_get(g.as_ptr(), ffi::PyUnicode_InternFromString(c"bound_wrapper".as_ptr()));
        let mut torch = HashSet::new();
        if ran.is_ok() && !found.is_null() && ffi::Py_TYPE(found) == &raw mut ffi::PyList_Type {
            for i in 0..ffi::PyList_GET_SIZE(found) {
                torch.insert(key(ffi::PyList_GET_ITEM(found, i)));
            }
            // The set refers to its objects by address: kept for good.
            std::mem::forget(g.unbind());
        }
        ffi::PyErr_Clear();
        // (The type of partials is looked up on first use: here.)
        tr_partial_func(ffi::Py_None());
        // (A type of the interpreter's: it lives as long as the process.)
        Made { torch, call: ffi::PyUnicode_InternFromString(c"__call__".as_ptr()) as usize, bound_wrapper: bound_wrapper as usize }
    });
    let _ = MADE.set(made);
}

// The kind of the callee `f` (not NULL), as an index into KINDS.
unsafe fn kind(f: *mut PyObject) -> usize {
    let tp = ffi::Py_TYPE(f);
    let made = MADE.get();
    let c_function = tp == &raw mut ffi::PyCFunction_Type;
    let descriptor = tp == &raw mut ffi::PyMethodDescr_Type || tp == &raw mut ffi::PyClassMethodDescr_Type;
    let wrapper = tp == &raw mut ffi::PyWrapperDescr_Type || made.is_some_and(|m| m.bound_wrapper == tp as usize);
    if tp == &raw mut ffi::PyFunction_Type {
        0
    } else if tp == &raw mut PyMethod_Type {
        1
    } else if (c_function || descriptor || tp == &raw mut ffi::PyWrapperDescr_Type) && made.is_some_and(|m| m.torch.contains(&key(f))) {
        2
    } else if c_function {
        3
    } else if descriptor {
        4
    } else if wrapper {
        5
    } else if tp == &raw mut ffi::PyType_Type {
        6
    } else if ffi::PyType_GetFlags(tp) & ffi::Py_TPFLAGS_TYPE_SUBCLASS != 0 {
        // (A flag of the callee's own type object, which a metaclass has:
        // the callee is a class, and its type is not `type` itself.)
        7
    } else if made.is_some() && !tr_partial_func(f).is_null() {
        8
    } else {
        // What the class has under __call__, as the interpreter looks it up
        // for a call of an instance.
        let call = made.map_or(std::ptr::null_mut(), |m| tr_type_lookup(tp, m.call as *mut PyObject));
        if call.is_null() {
            11
        } else if ffi::Py_TYPE(call) == &raw mut ffi::PyFunction_Type {
            9
        } else if ffi::Py_TYPE(call) == &raw mut ffi::PyWrapperDescr_Type {
            10
        } else {
            11
        }
    }
}

// What the table answers for the callee `f` that no family has a line for:
// the family of rules it is left to, or a stop.
pub unsafe fn of(f: *mut PyObject) -> Taken {
    KINDS[kind(f)].1
}

// The name of the kind of `f`, for the doors of tests.
pub unsafe fn kind_name(f: *mut PyObject) -> &'static str {
    KINDS[kind(f)].0
}
