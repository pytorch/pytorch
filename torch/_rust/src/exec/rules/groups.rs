// The groups of classes a row of the rule table is keyed on or constrains an
// operand by.
//
// A group is a named predicate over the EXACT type object of an operand: it
// never asks isinstance, and a subclass of a member is not a member. The
// enum is closed: a row can name no class that is not written down here, and
// a class no group names is in no group, so no row takes it.
//
// What it is given
//   - a type object by address (never NULL: mod.rs asks about no NULL
//     operand).
// What it guarantees
//   - holds() compares addresses only: it runs no code and reads nothing of
//     the type but which type it is;
//   - Immutable is the list below and nothing else: classes whose instances
//     cannot change after they are made and whose C slots (hash, compare,
//     truth, length, repr) run no code of the program;
//   - an exact tuple is NOT immutable in this sense: it is a container, and
//     what it holds may change or be of a class with code of its own.
// What it may not assume
//   - that a group says anything about the instance but its class: an exact
//     list may hold anything.
//
// The groups that follow CPython's attribute lookup (an instance dict, a
// class attribute, a descriptor, a `__getattr__`) are not here yet: they are
// computed from a type's slots and MRO and come with the rows of attribute
// loads, as more members of this enum.

use std::ffi::CStr;

use pyo3::ffi;
use pyo3::ffi::PyObject;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Group {
    Immutable,
    ExactList,
    ExactTuple,
    ExactDict,
    ExactSet,
    // Exactly torch.Tensor or nn.Parameter: the classes whose Python
    // operators run as plain aten operators.
    PlainTensor,
}

pub const GROUPS: &[Group] = &[Group::Immutable, Group::ExactList, Group::ExactTuple, Group::ExactDict, Group::ExactSet, Group::PlainTensor];

// The immutable classes, each with why it is on the list.
unsafe fn immutable() -> [(*mut ffi::PyTypeObject, &'static str); 7] {
    [
        (&raw mut ffi::PyLong_Type, "an int is its digits; its arithmetic, hash and comparison are C code over them"),
        (&raw mut ffi::PyFloat_Type, "a float is one C double"),
        (&raw mut ffi::PyBool_Type, "True and False are the only two bools, and bool has no subclass"),
        (&raw mut ffi::PyUnicode_Type, "a str is its characters; its hash is cached from them"),
        (&raw mut ffi::PyBytes_Type, "a bytes is its bytes"),
        (ffi::Py_TYPE(ffi::Py_None()), "None is the one instance of its class"),
        (ffi::Py_TYPE(ffi::Py_Ellipsis()), "... is the one instance of its class"),
    ]
}

impl Group {
    // Whether the class at `tp` (not NULL) is a member.
    pub unsafe fn holds(self, tp: *mut ffi::PyTypeObject) -> bool {
        match self {
            Group::Immutable => immutable().iter().any(|(member, _)| *member == tp),
            Group::ExactList => tp == &raw mut ffi::PyList_Type,
            Group::ExactTuple => tp == &raw mut ffi::PyTuple_Type,
            Group::ExactDict => tp == &raw mut ffi::PyDict_Type,
            Group::ExactSet => tp == &raw mut ffi::PySet_Type,
            Group::PlainTensor => super::super::plain_tensor_types().contains(&(tp as usize)),
        }
    }

    // What the dump says of the group: a line per member of the one group
    // that is a list, a sentence for each of the others.
    pub unsafe fn sentences(self) -> Vec<(String, &'static str)> {
        let one = |why: &'static str| vec![(format!("{self:?}"), why)];
        match self {
            Group::Immutable => immutable().iter().map(|(tp, why)| (format!("Immutable {}", std::ffi::CStr::from_ptr((**tp).tp_name).to_string_lossy()), *why)).collect(),
            Group::ExactList => one("exactly list: its items are read and written in C, and may be anything"),
            Group::ExactTuple => one("exactly tuple: its items are read in C and never change, and may be anything"),
            Group::ExactDict => one("exactly dict: a lookup is C code but for the hash and comparison of its keys"),
            Group::ExactSet => one("exactly set: a lookup is C code but for the hash and comparison of its members"),
            Group::PlainTensor => one("exactly torch.Tensor or torch.nn.Parameter: no __torch_function__ runs for it"),
        }
    }
}

// The class of an operand: exact, by the address of its type object (0: the
// operand is NULL), or any member of a group.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Class {
    Exact(usize),
    Group(Group),
}

impl Class {
    pub unsafe fn of(o: *mut PyObject) -> Class {
        Class::Exact(if o.is_null() { 0 } else { ffi::Py_TYPE(o) as usize })
    }

    // Whether a row that wants this class takes an operand of class `have`.
    pub(super) unsafe fn takes(self, have: Class) -> bool {
        match (self, have) {
            (Class::Exact(want), Class::Exact(tp)) => tp != 0 && want == tp,
            (Class::Group(g), Class::Exact(tp)) => tp != 0 && g.holds(tp as *mut ffi::PyTypeObject),
            (Class::Group(g), Class::Group(h)) => g == h,
            (Class::Exact(_), Class::Group(_)) => false,
        }
    }

    pub(super) unsafe fn name(self) -> String {
        match self {
            Class::Exact(0) => "NULL".into(),
            Class::Exact(tp) => CStr::from_ptr((*(tp as *mut ffi::PyTypeObject)).tp_name).to_string_lossy().into_owned(),
            Class::Group(g) => format!("{g:?}"),
        }
    }
}
