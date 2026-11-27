// The lines of a truth test: ToBool (the test of a conditional jump, of
// `and`, `or`, `not` on the way to one) and UNARY_NOT, by the class of the
// one operand.
//
// The interpreter's truth test (PyObject_IsTrue) is: True, False and None by
// identity; else the type's slot for truth; else its slot for a length;
// else true. The lines follow that, one group of exact classes each
// (groups.rs), and a class in none of the groups has no line: the test
// stops a trace.
//
// What it is given
//   - by rules::license, through the `before` of a row whose class took the
//     operand: the operand, before the test runs.
// What it guarantees
//   - a row is for a class whose truth runs C code of the interpreter over
//     the operand alone (an immutable, an exact tuple, an exact container, a
//     subclass of a builtin with the builtin's own truth), no code at all (a
//     class with no truth of its own), or the one Python function the class
//     has under `__bool__`, which the row names (enters()) and which the
//     interpreter itself holds to returning a bool;
//   - a truth through a Python `__len__` has NO row: what that function
//     returns is converted to a size through the `__index__` of its class,
//     which is code no row can name before the function ran. It is left, by
//     name, to the code that decided before the table;
//   - what a row's truth depends on besides the operand's own value is
//     handed to exec/reads.rs before the test runs: the length of a
//     container; WHICH CLASS THE OPERAND IS OF, for every operand that can
//     be given another class (an instance of a class that can change, and a
//     module: reads.rs's special()), with what that class has under
//     `__bool__` and `__len__`. A truth test writes nothing, so what is
//     recorded before it is what it finds;
//   - nothing here runs code of the program or makes anything on first use.
// What it may not assume
//   - that the frame a row names is the only code under the test: what that
//     function calls is asked about one instruction at a time, and its own
//     reads are recorded as its frame runs;
//   - that a row is all the license today's trace needs: the checks that
//     were there before the rows still run behind them.

use pyo3::ffi;
use pyo3::ffi::PyObject;

use super::super::reads;
use super::groups::{truth_function, truth_names};
use super::{Before, Class, Enters, Group, Line, Read, Res, Row, Says, Scalars, Site, What, Write};

// What every row of a truth test says: it writes nothing, and no scalar
// among its operands is anything but the operand tested. A line that names
// no class of its own takes no operand: no object is of the class that
// stands for "none".
const ROW: Row = Row { operands: &[(0, Class::Exact(0))], reads: &[], writes: Write::None, result: Res::Lit, enters: Enters::Never, scalars: Scalars::None, why: "", before: None, form: "" };

const TENSOR: &[(i8, Class)] = &[(0, Class::Group(Group::PlainTensor))];
const STAND_IN: &[(i8, Class)] = &[(0, Class::Group(Group::StandIn))];
const IMMUTABLE: &[(i8, Class)] = &[(0, Class::Group(Group::Immutable))];
const TUPLE: &[(i8, Class)] = &[(0, Class::Group(Group::ExactTuple))];
const IMMUTABLE_SIZED: &[(i8, Class)] = &[(0, Class::Group(Group::ImmutableSized))];
const FIXED: &[(i8, Class)] = &[(0, Class::Group(Group::FixedSubclass))];
const ORDERED: &[(i8, Class)] = &[(0, Class::Group(Group::ExactOrderedDict))];
const DICT: &[(i8, Class)] = &[(0, Class::Group(Group::ExactDict))];
const LIST: &[(i8, Class)] = &[(0, Class::Group(Group::ExactList))];
const SET: &[(i8, Class)] = &[(0, Class::Group(Group::ExactSet))];
const SIZED: &[(i8, Class)] = &[(0, Class::Group(Group::SizedSubclass))];
const PYTHON: &[(i8, Class)] = &[(0, Class::Group(Group::PythonTruth))];
const LENGTH: &[(i8, Class)] = &[(0, Class::Group(Group::PythonLength))];
const NONE: &[(i8, Class)] = &[(0, Class::Group(Group::NoTruth))];

const WHY_LENGTH: &str = "The truth of an exact container is whether it holds anything: its length, read in C. The length is what a region is checked on. The result is known by the length that was recorded.";

// In the order they are tried. (A plain tensor and the stand-in come first:
// a class in two groups meets the line of the first.)
pub const LINES: &[Line] = &[
    Line::Stop(TENSOR, "the truth of a tensor is a value of its data, and no check of a region reads data"),
    Line::Later(STAND_IN, "recorder"),
    Line::Row(Row { operands: IMMUTABLE, why: "The truth of a bool, None, an int, a float, a str, a bytes or the ellipsis is a function of its value, computed in C: nothing else is read. The result is known when the value of the operand is known.", ..ROW }),
    Line::Row(Row { operands: TUPLE, reads: &[Read::Identity], why: concat!("The truth of an exact tuple is whether it holds anything. ", "It holds what it held when it was made: the row records nothing, and what gave the operand (a global, an attribute, an input, a build of the run's own) says which object it is. The result is known when it is known which object the operand is, or when the run itself built it."), ..ROW }),
    Line::Row(Row { operands: IMMUTABLE_SIZED, reads: &[Read::Identity], why: concat!("The truth of an exact frozenset or range is whether it holds anything, read in C. ", "It holds what it held when it was made: the row records nothing, and what gave the operand (a global, an attribute, an input, a build of the run's own) says which object it is. The result is known when it is known which object the operand is, or when the run itself built it."), ..ROW }),
    Line::Row(Row { operands: FIXED, reads: &[Read::Class, Read::Identity], why: concat!("The class is a subclass of tuple, int or str that has under __bool__ and __len__ what the builtin has: the truth is the builtin's own C code over the operand's value (a tuple's length, an int's or a str's value). For a class that can change, which class the operand is of and what the class has under both names are what a region is checked on. ", "It holds what it held when it was made: the row records nothing, and what gave the operand (a global, an attribute, an input, a build of the run's own) says which object it is. The result is known when it is known which object the operand is, or when the run itself built it."), before: Some(fixed), ..ROW }),
    Line::Row(Row { operands: ORDERED, reads: &[Read::Length], why: WHY_LENGTH, before: Some(length), ..ROW }),
    Line::Row(Row { operands: DICT, reads: &[Read::Length], why: WHY_LENGTH, before: Some(length), ..ROW }),
    Line::Row(Row { operands: LIST, reads: &[Read::Length], why: WHY_LENGTH, before: Some(length), ..ROW }),
    Line::Row(Row { operands: SET, reads: &[Read::Length], why: WHY_LENGTH, before: Some(length), ..ROW }),
    Line::Row(Row { operands: SIZED, reads: &[Read::Class, Read::Length], why: "The class is a subclass of dict, list or set that has under __bool__ and __len__ what the builtin has: the truth is whether the container holds anything, the builtin's own length, read in C. The length and, for a class that can change, which class the operand is of and what the class has under both names are what a region is checked on. The result is known by the length that was recorded.", before: Some(sized), ..ROW }),
    Line::Row(Row { operands: PYTHON, reads: &[Read::Class], result: Res::Return, enters: Enters::Named, why: "The class has an exact Python function under __bool__: the test enters that function, once, and the result is what it returns, which the interpreter holds to being a bool. Which class the operand is of and what the class has under __bool__ and __len__ are what a region is checked on; what the function reads is recorded as its frame runs.", before: Some(python_truth), ..ROW }),
    Line::Later(LENGTH, "subscripts"),
    Line::Row(Row { operands: NONE, reads: &[Read::Class], why: "The class has nothing under __bool__ and __len__ and no slot for a truth or a length: every instance is true, and nothing of the operand is read. For an operand that can be given another class (an instance of a class that can change; a module), which class it is of and, for a class that can change, that the class has nothing under both names are what a region is checked on. The result is known: true.", before: Some(no_truth), ..ROW }),
];

// What the rows' `before` functions refuse and record, for the dump.
pub const BEFORES: &[Says] = &[
    (fixed as Before, "truth_fixed", "Refuses nothing. Records which class the operand is of and what that class has under __bool__ and __len__, for an operand of a class that can change (reads::truth)."),
    (length as Before, "truth_length", "Refuses nothing. Records the length of the container (reads::truth)."),
    (sized as Before, "truth_sized", "Refuses nothing. Records the length of the container (reads::truth) and, for an operand of a class that can change, which class it is of and what that class has under __bool__ and __len__ (reads::class_of)."),
    (python_truth as Before, "truth_python", "Refuses nothing. Records which class the operand is of and what that class has under __bool__ and __len__ (reads::truth); names the function the test enters (rules::names)."),
    (no_truth as Before, "truth_none", "Refuses nothing. Records which class the operand is of, for an instance of a class that can change and for a module, and for a class that can change that it has nothing under __bool__ and __len__ (reads::truth); nothing for an operand that cannot be given another class."),
];

// What a row's `before` records. Each row's function hands its own tag to
// the one body below, so that no two of them are one piece of code: a row
// is told by its function (the dump, names()), and two functions with one
// body may be one function to the compiler.
#[derive(Copy, Clone, PartialEq, Eq)]
enum Tag {
    Fixed,
    Length,
    Sized,
    Function,
    None,
}

// Hands the operand to the hook the executor calls behind a truth test,
// which records by what the operand is: a container's length; else, for an
// instance of a class that can change, its class and what the class has
// under both names, and for a module its class. A subclass of dict, list or
// set is a container to that hook, which records its length alone: the
// class and the two lookups are recorded here.
unsafe fn record(tag: Tag, site: &Site) -> Result<(), String> {
    let Some(&operand) = site.operands.first() else { return Err(format!("it has no operand ({})", tag as u8)) };
    if tag == Tag::Sized {
        for name in truth_names() {
            if !name.is_null() {
                reads::class_of(operand, name);
            }
        }
    }
    reads::truth(operand);
    Ok(())
}

unsafe fn fixed(_: &Row, _: &What, site: &Site) -> Result<(), String> {
    record(Tag::Fixed, site)
}

unsafe fn length(_: &Row, _: &What, site: &Site) -> Result<(), String> {
    record(Tag::Length, site)
}

unsafe fn sized(_: &Row, _: &What, site: &Site) -> Result<(), String> {
    record(Tag::Sized, site)
}

pub unsafe fn python_truth(_: &Row, _: &What, site: &Site) -> Result<(), String> {
    record(Tag::Function, site)
}

unsafe fn no_truth(_: &Row, _: &What, site: &Site) -> Result<(), String> {
    record(Tag::None, site)
}

// The function a truth test of the operand enters, for the row of
// PythonTruth: what the operand's class has under `__bool__`.
pub unsafe fn enters(_: &Row, _: &What, site: &Site) -> [*mut PyObject; 2] {
    let object: *mut PyObject = site.operands.first().copied().unwrap_or(std::ptr::null_mut());
    [if object.is_null() { std::ptr::null_mut() } else { truth_function(ffi::Py_TYPE(object)) }, std::ptr::null_mut()]
}
