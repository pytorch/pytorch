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
// THE GROUPS OF A TRUTH TEST follow what PyObject_IsTrue does with an object
// of the type: the type's own slot for truth, else its slot for a length,
// else true. For a class written in Python those slots look `__bool__`,
// then `__len__`, up on the class (slot_nb_bool), so the group is told by
// that same lookup (_PyType_Lookup, in C): PythonTruth where it finds an
// exact Python function under `__bool__`; PythonLength where it finds
// nothing there and an exact Python function under `__len__`; NoTruth where
// it finds nothing under either name and the three slots of the type are
// empty; SizedSubclass and FixedSubclass where it finds under both names
// exactly what the builtin class does that the class is a subclass of (the
// builtin's own C slots run). A class that is in none of them (a subclass of
// float, a C type with a truth of its own, a class with something that is
// no function under `__bool__`) is in no group of truth. Nor is a metaclass
// written in Python (its instances are classes: the executor's hook for a
// truth test records no lookup for an operand that is a class).
//
// EVERY GROUP THAT ready() MAKES holds nothing before ready() ran, and
// nothing at all if ready() could not find what it looks for (made()): a
// class is then in no group that is told by a name or by a class of a
// module, and an instruction on it has no line.
//
// The groups that follow CPython's attribute lookup (an instance dict, a
// class attribute, a descriptor, a `__getattr__`) are not here yet: they are
// computed from a type's slots and MRO and come with the rows of attribute
// loads, as more members of this enum.

use std::ffi::CStr;
use std::sync::OnceLock;

use pyo3::ffi;
use pyo3::ffi::PyObject;
use pyo3::prelude::*;

use super::super::ffi::{tr_type_heap, tr_type_lookup};

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
    ExactModule,
    // Exactly torch.dtype, device, layout, memory_format or Size: values of
    // torch's own C classes that cannot change.
    TorchConstant,
    ExactOrderedDict,
    // Exactly the class of the engine's stand-in for a number that is an
    // input of a region (a subclass of float of the tracer's).
    StandIn,
    // A class written in Python, no metaclass and no subclass of dict,
    // list or set, whose truth is the exact Python function it has under
    // `__bool__`.
    PythonTruth,
    // A class written in Python, no metaclass and no subclass of dict,
    // list or set, that has nothing under `__bool__` and an exact Python
    // function under `__len__`.
    PythonLength,
    // Exactly frozenset or range: what it holds is fixed when it is made.
    ImmutableSized,
    // A subclass of tuple, int or str (not bool) whose truth is the
    // builtin's own: it has under `__bool__` and `__len__` what the builtin
    // has.
    FixedSubclass,
    // A subclass of dict, list or set (not exactly OrderedDict) whose truth
    // is the builtin's own length: it has under `__bool__` and `__len__`
    // what the builtin has.
    SizedSubclass,
    // A class whose instances are all true: nothing under `__bool__` and
    // `__len__`, and no slot for a truth or a length; no metaclass written
    // in Python.
    NoTruth,
}

pub const GROUPS: &[Group] = &[
    Group::Immutable,
    Group::ExactList,
    Group::ExactTuple,
    Group::ExactDict,
    Group::ExactSet,
    Group::PlainTensor,
    Group::ExactModule,
    Group::TorchConstant,
    Group::ExactOrderedDict,
    Group::StandIn,
    Group::PythonTruth,
    Group::PythonLength,
    Group::ImmutableSized,
    Group::FixedSubclass,
    Group::SizedSubclass,
    Group::NoTruth,
];

// What the groups compare with that is not a type of the interpreter's C
// API: found once, by ready().
struct Made {
    ordered_dict: usize,
    stand_in: usize,
    // `__bool__` and `__len__`, interned.
    truth: usize,
    len: usize,
}

static MADE: OnceLock<Made> = OnceLock::new();

// The two classes, by a snippet of Python that names them: `ordered` and
// `stand_in` in its namespace.
const CLASSES: &CStr = c"
import collections
ordered = collections.OrderedDict
from torch._rust_inductor import _Scalar as stand_in
";

// What ready() made, if it found all of it.
fn made() -> Option<&'static Made> {
    MADE.get().filter(|m| m.ordered_dict != 0 && m.stand_in != 0 && m.truth != 0 && m.len != 0)
}

// Finds the classes and names above. Runs Python code (the snippet): called
// where a function goes onto the executor, never under an instruction
// (rules::ready). Before it ran, and if it did not find every one of them,
// no class is in a group that needs any of them (made()).
pub fn ready() {
    if MADE.get().is_some() {
        return;
    }
    let made = Python::attach(|py| unsafe {
        let g = pyo3::types::PyDict::new(py);
        let ran = py.run(CLASSES, Some(&g), None);
        let class = |name: &CStr| if ran.is_ok() { super::super::reads::dict_get(g.as_ptr(), ffi::PyUnicode_InternFromString(name.as_ptr())) as usize } else { 0 };
        // (Classes of modules that stay loaded: they live as long as the
        // process, as the interned names do.)
        let made = Made {
            ordered_dict: class(c"ordered"),
            stand_in: class(c"stand_in"),
            truth: ffi::PyUnicode_InternFromString(c"__bool__".as_ptr()) as usize,
            len: ffi::PyUnicode_InternFromString(c"__len__".as_ptr()) as usize,
        };
        ffi::PyErr_Clear();
        made
    });
    let _ = MADE.set(made);
}

// What the class at `tp` has under `__bool__` and under `__len__`, as
// slot_nb_bool looks them up (borrowed, NULL: nothing). None unless ready()
// made its names.
unsafe fn truth_lookups(tp: *mut ffi::PyTypeObject) -> Option<[*mut PyObject; 2]> {
    let made = made()?;
    Some([tr_type_lookup(tp, made.truth as *mut PyObject), tr_type_lookup(tp, made.len as *mut PyObject)])
}

unsafe fn function(found: *mut PyObject) -> bool {
    !found.is_null() && ffi::Py_TYPE(found) == &raw mut ffi::PyFunction_Type
}

// The Python function that a truth test of an instance of the class at `tp`
// enters (borrowed), if the class is in PythonTruth; else NULL.
pub unsafe fn truth_function(tp: *mut ffi::PyTypeObject) -> *mut PyObject {
    if Group::PythonTruth.holds(tp) { truth_lookups(tp).map_or(std::ptr::null_mut(), |found| found[0]) } else { std::ptr::null_mut() }
}

// The two names a truth test looks up on a class, interned (NULL unless
// ready() made them): for a row that records those lookups itself.
pub fn truth_names() -> [*mut PyObject; 2] {
    made().map_or([std::ptr::null_mut(); 2], |m| [m.truth as *mut PyObject, m.len as *mut PyObject])
}

// Whether the class at `tp` is a subclass, not the class itself, of one of
// `builtins` and has under `__bool__` and `__len__` exactly what that
// builtin has: its truth is the builtin's own C code.
unsafe fn builtin_truth(tp: *mut ffi::PyTypeObject, builtins: &[*mut ffi::PyTypeObject]) -> bool {
    let Some(have) = truth_lookups(tp) else { return false };
    builtins.iter().any(|base| tp != *base && ffi::PyType_IsSubtype(tp, *base) != 0 && truth_lookups(*base) == Some(have))
}

// Whether the type has no slot that PyObject_IsTrue asks.
unsafe fn no_truth_slot(tp: *mut ffi::PyTypeObject) -> bool {
    let number = (*tp).tp_as_number;
    let mapping = (*tp).tp_as_mapping;
    let sequence = (*tp).tp_as_sequence;
    (number.is_null() || (*number).nb_bool.is_none()) && (mapping.is_null() || (*mapping).mp_length.is_none()) && (sequence.is_null() || (*sequence).sq_length.is_none())
}

// Whether the class is dict, list or set or a subclass of one: the classes
// whose truth the executor's hook records as a length.
unsafe fn sized_builtin(tp: *mut ffi::PyTypeObject) -> bool {
    ffi::PyType_GetFlags(tp) & (ffi::Py_TPFLAGS_DICT_SUBCLASS | ffi::Py_TPFLAGS_LIST_SUBCLASS) != 0 || ffi::PyType_IsSubtype(tp, &raw mut ffi::PySet_Type) != 0
}

// Whether the class is a metaclass written in Python: its instances are
// classes, for which the executor's hook records no lookup.
unsafe fn python_metaclass(tp: *mut ffi::PyTypeObject) -> bool {
    tr_type_heap(tp) != 0 && ffi::PyType_GetFlags(tp) & ffi::Py_TPFLAGS_TYPE_SUBCLASS != 0
}

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
            Group::ExactModule => tp == &raw mut ffi::PyModule_Type,
            Group::TorchConstant => super::super::torch_constant_types().contains(&(tp as usize)),
            Group::ExactOrderedDict => made().is_some_and(|m| m.ordered_dict == tp as usize),
            Group::StandIn => made().is_some_and(|m| m.stand_in == tp as usize),
            Group::PythonTruth => tr_type_heap(tp) != 0 && !python_metaclass(tp) && !sized_builtin(tp) && truth_lookups(tp).is_some_and(|found| function(found[0])),
            Group::PythonLength => tr_type_heap(tp) != 0 && !python_metaclass(tp) && !sized_builtin(tp) && truth_lookups(tp).is_some_and(|found| found[0].is_null() && function(found[1])),
            Group::ImmutableSized => tp == &raw mut ffi::PyFrozenSet_Type || tp == &raw mut ffi::PyRange_Type,
            Group::FixedSubclass => tp != &raw mut ffi::PyBool_Type && !Group::StandIn.holds(tp) && builtin_truth(tp, &[&raw mut ffi::PyTuple_Type, &raw mut ffi::PyLong_Type, &raw mut ffi::PyUnicode_Type]),
            Group::SizedSubclass => !Group::ExactOrderedDict.holds(tp) && builtin_truth(tp, &[&raw mut ffi::PyDict_Type, &raw mut ffi::PyList_Type, &raw mut ffi::PySet_Type]),
            Group::NoTruth => !python_metaclass(tp) && truth_lookups(tp).is_some_and(|found| found[0].is_null() && found[1].is_null()) && no_truth_slot(tp),
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
            Group::ExactModule => one("exactly types.ModuleType: an attribute of it is the item of its dict, else what the __getattr__ of that dict returns"),
            Group::TorchConstant => one("exactly torch.dtype, torch.device, torch.layout, torch.memory_format or torch.Size: a value of a C class of torch that cannot change, whose hash, comparison and repr are C code (a Size is a tuple of ints)"),
            Group::ExactOrderedDict => one("exactly collections.OrderedDict: a dict of a C class whose length is the dict's own; its other slots are C code of its own"),
            Group::StandIn => one("exactly the class of the engine's stand-in for a number that is an input of a region: a subclass of float that the tracer of today's engine puts in the program's slots"),
            Group::PythonTruth => one("a class written in Python, no metaclass and no subclass of dict, list or set, that has an exact Python function under __bool__: a truth test of an instance enters that function, which is to return a bool"),
            Group::PythonLength => one("a class written in Python, no metaclass and no subclass of dict, list or set, that has nothing under __bool__ and an exact Python function under __len__: a truth test of an instance enters that function and converts what it returns to a size, through the __index__ of the class of what it returns"),
            Group::ImmutableSized => one("exactly frozenset or range: what it holds is fixed when it is made, and its length is read in C"),
            Group::FixedSubclass => one("a subclass of tuple, int or str, not bool, that has under __bool__ and under __len__ exactly what that builtin has (a namedtuple, a member of an IntEnum, torch.Size): its truth is the builtin's own C code over a value that never changes"),
            Group::SizedSubclass => one("a subclass of dict, list or set, not exactly collections.OrderedDict, that has under __bool__ and under __len__ exactly what that builtin has (a defaultdict, a Counter, a dict subclass with no truth of its own): its truth is the builtin's own length"),
            Group::NoTruth => one("a class, no metaclass written in Python, that has nothing under __bool__ and under __len__ and no slot for a truth or a length: every instance is true, and nothing of the instance is read"),
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
