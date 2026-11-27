// The two tables of a slot callee (binary, compare, unary, the subscripts,
// the attributes), each written here and nowhere else, and which objects a
// part is read off. Nothing here runs code of the program.
//
// LICENSE: which slot callees may be entries. `_calls_make` takes one only
// if the pinned classes of ALL its operands are a line; no line is a
// refusal. A line names the callee's kind, its operators (or attribute
// names) and, per operand, a closed predicate over exact classes (`Is`,
// built on the groups of the rule table). Its sentence says why C code alone
// runs for such operands; a test runs every line and every operator of it
// under a profiler, in the plain program and in a list. What is in no line:
// a subclass of Tensor, a class written in Python, an nn.Module, a dict as
// the container (a stored key of a Python class with the hash of the key
// looked up has its __eq__ called: no fact of a class), a store to an
// attribute, a tensor attribute whose getter can warn (T, grad) or that the
// sweep of torch's members has no line for. The rows of the rule table widen
// this.
// What no line can say, because it is no fact of a class: a kernel of aten
// may issue a WARNING for operands of some dtype or value (an index of dtype
// uint8; a complex value stored into a real tensor), and a warning runs the
// warnings module, in the plain program and under a list alike. That is the
// "P = warn" of a member's sweep line, for the callee's row to say.
//
// The table does not stand alone. Behind it csrc/calls_slots.c resolves, from
// the same classes, what the slot would run, one step into CPython's
// dispatcher (a line's sentence is about torch.Tensor as torch defines it;
// a method assigned to the class since is seen there), and both must agree:
// a line that holds with a slot that would run a Python method is a refusal.
// That check is stricter than CPython in one place: for operands of two
// classes it wants the reflected method to be C code too, though the forward
// one answers first (Parameter - Tensor is refused: Tensor.__rsub__ is a
// Python function).
//
// ORDER: in which order an entry releases the operands it moves and what it
// built, per kind, as CPython's instruction of that kind does. Each line was
// read off the plain statement run with operands whose death is logged; a
// test holds every line against that log and against a list.

use std::ffi::{c_char, c_int, CStr};
use std::ptr::addr_of_mut;

use pyo3::ffi::{self, PyObject, PyTypeObject};

use super::{Callee, Tmpl, UNARY_OPS};
use crate::cfg::ir::{BINARY_OPS, COMPARE_OPS};
use crate::exec::rules::Group;

unsafe extern "C" {
    fn tr_slot_python(kind: c_int, op: c_int, a: *mut PyTypeObject, b: *mut PyTypeObject, name: *mut PyObject) -> *const c_char;
    fn tr_tuple_like(tp: *mut PyTypeObject) -> c_int;
}

// What the license sees of an operand: the pinned class of a leaf, the parts
// of what the entry builds. A constant that is an exact tuple is its items
// (a constant is held by the list and a tuple never changes).
pub enum Shape {
    Null,
    Leaf(*mut PyTypeObject),
    Slice(Vec<Shape>),
    Tuple(Vec<Shape>),
    List(Vec<Shape>),
    Map(Vec<Shape>),
}

// What a line says of one operand.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Is {
    Tensor,
    Number,
    Int,
    Str,
    Bytes,
    List,
    Sequence,
    Stored,
    IntSlice,
    Scalar,
    Index,
}

pub const EVERY_IS: [Is; 11] = [Is::Tensor, Is::Number, Is::Int, Is::Str, Is::Bytes, Is::List, Is::Sequence, Is::Stored, Is::IntSlice, Is::Scalar, Is::Index];

unsafe fn leaf(s: &Shape, classes: &[*mut PyTypeObject]) -> bool {
    matches!(s, Shape::Leaf(tp) if classes.contains(tp))
}

unsafe fn none() -> *mut PyTypeObject {
    unsafe { ffi::Py_TYPE(ffi::Py_None()) }
}

impl Is {
    pub fn sentence(self) -> &'static str {
        match self {
            Is::Tensor => "exactly torch.Tensor or torch.nn.Parameter (the group PlainTensor)",
            Is::Number => "exactly int, float or bool",
            Is::Int => "exactly int",
            Is::Str => "exactly str",
            Is::Bytes => "exactly bytes",
            Is::List => "exactly list: one that is pinned, or one the entry builds",
            Is::Sequence => "exactly list or tuple: pinned, or built by the entry",
            Is::Stored => "an operand that is only stored: of an immutable class, a plain tensor, or exactly a list, tuple, dict or set",
            Is::IntSlice => "a slice the entry builds whose three bounds are each exactly int, None or absent",
            Is::Scalar => "a plain tensor, or exactly int, float or bool",
            Is::Index => "exactly int, None, Ellipsis, a plain tensor, a slice the entry builds whose bounds are each exactly int, None, absent or a plain tensor, or an exact tuple of those (built by the entry, or a constant)",
        }
    }

    pub unsafe fn holds(self, s: &Shape) -> bool {
        unsafe {
            let (int, float, boolean) = (addr_of_mut!(ffi::PyLong_Type), addr_of_mut!(ffi::PyFloat_Type), addr_of_mut!(ffi::PyBool_Type));
            let (list, tuple) = (addr_of_mut!(ffi::PyList_Type), addr_of_mut!(ffi::PyTuple_Type));
            let tensor = |s: &Shape| matches!(s, Shape::Leaf(tp) if Group::PlainTensor.holds(*tp));
            let bounds = |s: &Shape, also_tensor: bool| match s {
                Shape::Slice(parts) => parts.iter().all(|p| matches!(p, Shape::Null) || leaf(p, &[int, none()]) || (also_tensor && tensor(p))),
                Shape::Null | Shape::Leaf(_) | Shape::Tuple(_) | Shape::List(_) | Shape::Map(_) => false,
            };
            let one = |s: &Shape| leaf(s, &[int, none(), ffi::Py_TYPE(ffi::Py_Ellipsis())]) || tensor(s) || bounds(s, true);
            match self {
                Is::Tensor => tensor(s),
                Is::Number => leaf(s, &[int, float, boolean]),
                Is::Int => leaf(s, &[int]),
                Is::Str => leaf(s, &[addr_of_mut!(ffi::PyUnicode_Type)]),
                Is::Bytes => leaf(s, &[addr_of_mut!(ffi::PyBytes_Type)]),
                Is::List => leaf(s, &[list]) || matches!(s, Shape::List(_)),
                Is::Sequence => leaf(s, &[list, tuple]) || matches!(s, Shape::List(_) | Shape::Tuple(_)),
                Is::Stored => match s {
                    Shape::Leaf(tp) => [Group::Immutable, Group::PlainTensor, Group::ExactList, Group::ExactTuple, Group::ExactDict, Group::ExactSet].iter().any(|g| g.holds(*tp)),
                    Shape::Tuple(_) | Shape::List(_) | Shape::Map(_) => true,
                    Shape::Null | Shape::Slice(_) => false,
                },
                Is::IntSlice => bounds(s, false),
                Is::Scalar => tensor(s) || leaf(s, &[int, float, boolean]),
                Is::Index => match s {
                    Shape::Tuple(items) => items.iter().all(one),
                    Shape::Null | Shape::Leaf(_) | Shape::Slice(_) | Shape::List(_) | Shape::Map(_) => one(s),
                },
            }
        }
    }
}

pub struct Line {
    // What a test names the line by.
    pub name: &'static str,
    pub kind: &'static str,
    // The operators of a binary, compare or unary line; the names of an
    // attribute line; none for a subscript.
    pub ops: &'static [&'static str],
    // One per operand, in the order of the entry's operands.
    pub operands: &'static [Is],
    pub why: &'static str,
}

// The operators whose method of TensorBase is C code: forward and in place
// (not `//`, `**`, `**=`: Python functions of torch.Tensor; not `//=`: its
// method is C code but hands over to `//`), and reflected.
const TENSOR: &[&str] = &["+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>", "@", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=", "@="];
const TENSOR_NUMBER: &[&str] = &["+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="];
const REFLECTED: &[&str] = &["+", "*", "&", "|", "^", "+=", "*=", "&=", "|=", "^="];
const CONCAT: &[&str] = &["+", "+="];
const REPEAT: &[&str] = &["*", "*="];
// The getset descriptors of TensorBase that the sweep of torch's members has
// a line for (no Python frame, no write, no key change, no draw), less `T`
// (warns for a tensor of other than 2 dimensions) and `grad` (warns for a
// tensor that is no leaf): a warning runs the warnings module.
const TENSOR_FACTS: &[&str] = &["shape", "dtype", "device", "layout", "ndim", "requires_grad", "is_cpu", "is_cuda", "is_sparse", "is_nested", "mT", "mH", "H"];

pub const LICENSE: &[Line] = &[
    Line { name: "binary.tensor.tensor", kind: "binary", ops: TENSOR, operands: &[Is::Tensor, Is::Tensor], why: "TensorBase's method of the operator is C code that takes a tensor and answers a tensor or raises, and a plain tensor has no __torch_function__ that is called" },
    Line { name: "binary.tensor.number", kind: "binary", ops: TENSOR_NUMBER, operands: &[Is::Tensor, Is::Number], why: "the same C method takes an int, a float or a bool as a scalar; a number has no reflected method that is Python" },
    Line { name: "binary.number.tensor", kind: "binary", ops: REFLECTED, operands: &[Is::Number, Is::Tensor], why: "the number's C method answers NotImplemented for a tensor, and TensorBase's reflected method of these five operators is C code" },
    Line { name: "binary.number.number", kind: "binary", ops: &BINARY_OPS, operands: &[Is::Number, Is::Number], why: "int, float and bool are static types whose number methods are C functions over the two values" },
    Line { name: "binary.str.str", kind: "binary", ops: CONCAT, operands: &[Is::Str, Is::Str], why: "str's concatenation is C code over the characters of both" },
    Line { name: "binary.bytes.bytes", kind: "binary", ops: CONCAT, operands: &[Is::Bytes, Is::Bytes], why: "bytes' concatenation is C code over the bytes of both" },
    Line { name: "binary.str.int", kind: "binary", ops: REPEAT, operands: &[Is::Str, Is::Int], why: "str's repetition reads the count off the int in C" },
    Line { name: "binary.int.str", kind: "binary", ops: REPEAT, operands: &[Is::Int, Is::Str], why: "int's C method answers NotImplemented for a str, and str's repetition reads the count off the int in C" },
    Line { name: "binary.bytes.int", kind: "binary", ops: REPEAT, operands: &[Is::Bytes, Is::Int], why: "bytes' repetition reads the count off the int in C" },
    Line { name: "binary.int.bytes", kind: "binary", ops: REPEAT, operands: &[Is::Int, Is::Bytes], why: "int's C method answers NotImplemented for a bytes, and bytes' repetition reads the count off the int in C" },
    Line { name: "compare.tensor.tensor", kind: "compare", ops: &COMPARE_OPS, operands: &[Is::Tensor, Is::Tensor], why: "TensorBase's six comparison methods are C code that answers a tensor or raises" },
    Line { name: "compare.tensor.number", kind: "compare", ops: &COMPARE_OPS, operands: &[Is::Tensor, Is::Number], why: "the same C methods take an int, a float or a bool as a scalar" },
    Line { name: "compare.number.tensor", kind: "compare", ops: &COMPARE_OPS, operands: &[Is::Number, Is::Tensor], why: "the number's C comparison answers NotImplemented for a tensor, and the reflected comparison is one of TensorBase's six C methods" },
    Line { name: "compare.number.number", kind: "compare", ops: &COMPARE_OPS, operands: &[Is::Number, Is::Number], why: "int, float and bool compare in C over the two values" },
    Line { name: "compare.str.str", kind: "compare", ops: &COMPARE_OPS, operands: &[Is::Str, Is::Str], why: "str compares in C over the characters of both" },
    Line { name: "compare.bytes.bytes", kind: "compare", ops: &COMPARE_OPS, operands: &[Is::Bytes, Is::Bytes], why: "bytes compares in C over the bytes of both" },
    Line { name: "unary.tensor", kind: "unary", ops: &["-", "~", "+"], operands: &[Is::Tensor], why: "torch.Tensor holds TensorBase's C methods neg and positive under __neg__ and __pos__, and __invert__ is TensorBase's" },
    Line { name: "unary.number", kind: "unary", ops: &["-", "+"], operands: &[Is::Number], why: "int's, float's and bool's negation and positive are C functions over the value (not `~`: on a bool it warns, and a warning runs the warnings module)" },
    Line { name: "unary.int", kind: "unary", ops: &["~"], operands: &[Is::Int], why: "int's inversion is a C function over the value" },
    Line { name: "getitem.sequence.int", kind: "getitem", ops: &[], operands: &[Is::Sequence, Is::Int], why: "list's and tuple's subscript read the position off the int and hand out the item, in C" },
    Line { name: "getitem.sequence.slice", kind: "getitem", ops: &[], operands: &[Is::Sequence, Is::IntSlice], why: "the bounds are read off ints in C and the items are copied into a new list or tuple" },
    Line { name: "getitem.tensor", kind: "getitem", ops: &[], operands: &[Is::Tensor, Is::Index], why: "TensorBase's subscript takes ints, None, Ellipsis, slices and tensors in C; a tensor as a bound of a slice is read by TensorBase's C __index__" },
    Line { name: "setitem.list.int", kind: "setitem", ops: &[], operands: &[Is::Stored, Is::List, Is::Int], why: "list's item store puts the value in place of the item in C and reads nothing of the value" },
    Line { name: "setitem.list.slice", kind: "setitem", ops: &[], operands: &[Is::Sequence, Is::List, Is::IntSlice], why: "list's slice store takes the items of an exact list or tuple in C (anything else would be iterated)" },
    Line { name: "setitem.tensor", kind: "setitem", ops: &[], operands: &[Is::Scalar, Is::Tensor, Is::Index], why: "TensorBase's store takes the index as its subscript does and the value as a tensor or a scalar, in C" },
    Line { name: "delitem.list.int", kind: "delitem", ops: &[], operands: &[Is::List, Is::Int], why: "list's item deletion reads the position off the int in C" },
    Line { name: "delitem.list.slice", kind: "delitem", ops: &[], operands: &[Is::List, Is::IntSlice], why: "list's slice deletion reads the bounds off ints in C" },
    Line { name: "getattr.tensor", kind: "getattr", ops: TENSOR_FACTS, operands: &[Is::Tensor], why: "the name is a getset descriptor of TensorBase whose C getter reads the tensor's key or makes a view, and for a plain tensor calls no __torch_function__" },
];

pub struct Order {
    pub kind: &'static str,
    // The line of a getitem or setitem whose key is ("slice", a, b, ("null",)).
    pub two_bounds: bool,
    pub instruction: &'static str,
    // Operand positions; none: all, in their order.
    pub operands: &'static [usize],
    pub how: &'static str,
}

// An operand's release: the slots it moves, in the order of its leaves, then
// what the entry built for it. A container lets go of its parts as its type
// does: a list or tuple from the last to the first, a dict in order, a slice
// step, start, stop.
pub const ORDER: &[Order] = &[
    Order { kind: "vector", two_bounds: false, instruction: "CALL", operands: &[], how: "the operands in their order, positional then keyword values (the callee is held by identity: never a slot)" },
    Order { kind: "binary", two_bounds: false, instruction: "BINARY_OP", operands: &[0, 1], how: "left, right" },
    Order { kind: "compare", two_bounds: false, instruction: "COMPARE_OP", operands: &[0, 1], how: "left, right" },
    Order { kind: "unary", two_bounds: false, instruction: "UNARY_NEGATIVE, UNARY_INVERT, CALL_INTRINSIC_1", operands: &[0], how: "the operand" },
    Order { kind: "getitem", two_bounds: false, instruction: "BINARY_SUBSCR", operands: &[0, 1], how: "container, key" },
    Order { kind: "getitem", two_bounds: true, instruction: "BINARY_SLICE", operands: &[1, 0], how: "the slice (start, stop), container" },
    Order { kind: "setitem", two_bounds: false, instruction: "STORE_SUBSCR", operands: &[0, 1, 2], how: "value, container, key" },
    Order { kind: "setitem", two_bounds: true, instruction: "STORE_SLICE", operands: &[2, 0, 1], how: "the slice (start, stop), value, container" },
    Order { kind: "delitem", two_bounds: false, instruction: "DELETE_SUBSCR", operands: &[0, 1], how: "container, key (a slice of two bounds too: BUILD_SLICE made it)" },
    Order { kind: "getattr", two_bounds: false, instruction: "LOAD_ATTR", operands: &[0], how: "the owner" },
    Order { kind: "setattr", two_bounds: false, instruction: "STORE_ATTR", operands: &[0, 1], how: "value, owner (no line of the license takes a setattr: no list states it yet)" },
];

// The kind of a slot callee and its operator or attribute name; a vector is
// ("vector", ""). None: a state setter, a drop.
unsafe fn kind(callee: &Callee) -> Option<(&'static str, String)> {
    let text = |name: &pyo3::Py<pyo3::types::PyString>| unsafe {
        let mut n = 0;
        let p = ffi::PyUnicode_AsUTF8AndSize(name.as_ptr(), &mut n);
        if p.is_null() {
            ffi::PyErr_Clear();
            return String::new();
        }
        String::from_utf8_lossy(std::slice::from_raw_parts(p as *const u8, n as usize)).into_owned()
    };
    Some(match callee {
        Callee::Vector(..) => ("vector", String::new()),
        Callee::Binary(op) => ("binary", BINARY_OPS.get(*op as usize)?.to_string()),
        Callee::Compare(op) => ("compare", COMPARE_OPS.get(*op as usize)?.to_string()),
        Callee::Unary(opcode, oparg) => ("unary", UNARY_OPS.iter().find(|u| (u.1, u.2) == (*opcode, *oparg))?.0.to_string()),
        Callee::GetItem => ("getitem", String::new()),
        Callee::SetItem => ("setitem", String::new()),
        Callee::DelItem => ("delitem", String::new()),
        Callee::GetAttr(name) => ("getattr", text(name)),
        Callee::SetAttr(name) => ("setattr", text(name)),
        Callee::State(_) | Callee::Drop(_) => return None,
    })
}

// The operand positions of an entry of `callee` in the order its instruction
// releases them. None: no line of ORDER is of this callee and as many
// operands.
pub fn order(callee: &Callee, args: &[Tmpl]) -> Option<Vec<usize>> {
    let two_bounds = |key: usize| matches!(args.get(key), Some(Tmpl::Slice(p)) if matches!(p[2], Tmpl::Null));
    let (name, two) = match callee {
        Callee::GetItem => ("getitem", two_bounds(1)),
        Callee::SetItem => ("setitem", two_bounds(2)),
        Callee::Binary(_) => ("binary", false),
        Callee::Compare(_) => ("compare", false),
        Callee::Unary(..) => ("unary", false),
        Callee::DelItem => ("delitem", false),
        Callee::GetAttr(_) => ("getattr", false),
        Callee::SetAttr(_) => ("setattr", false),
        Callee::Vector(..) => ("vector", false),
        Callee::State(_) | Callee::Drop(_) => return None,
    };
    let line = ORDER.iter().find(|o| o.kind == name && o.two_bounds == two)?;
    if line.operands.is_empty() {
        return Some((0..args.len()).collect());
    }
    (line.operands.len() == args.len()).then(|| line.operands.to_vec())
}

// The class an operand is of as a whole.
unsafe fn class(s: &Shape) -> *mut PyTypeObject {
    match s {
        Shape::Null => std::ptr::null_mut(),
        Shape::Leaf(tp) => *tp,
        Shape::Slice(_) => addr_of_mut!(ffi::PySlice_Type),
        Shape::Tuple(_) => addr_of_mut!(ffi::PyTuple_Type),
        Shape::List(_) => addr_of_mut!(ffi::PyList_Type),
        Shape::Map(_) => addr_of_mut!(ffi::PyDict_Type),
    }
}

unsafe fn class_text(s: &Shape) -> String {
    unsafe {
        let all = |parts: &[Shape]| parts.iter().map(|p| class_text(p)).collect::<Vec<_>>().join(", ");
        match s {
            Shape::Null => "-".into(),
            Shape::Leaf(tp) => CStr::from_ptr((**tp).tp_name).to_string_lossy().rsplit('.').next().unwrap_or("?").to_string(),
            Shape::Slice(parts) => format!("slice({})", all(parts)),
            Shape::Tuple(parts) => format!("({})", all(parts)),
            Shape::List(parts) => format!("[{}]", all(parts)),
            Shape::Map(parts) => format!("{{{}}}", all(parts)),
        }
    }
}

unsafe fn second(kind: c_int, op: c_int, a: *mut PyTypeObject, b: *mut PyTypeObject, name: *mut PyObject) -> Result<(), String> {
    let why = unsafe { tr_slot_python(kind, op, a, b, name) };
    if why.is_null() {
        return Ok(());
    }
    Err(format!("a line of the license takes the operands, but the slots of their classes do not agree: {}", unsafe { CStr::from_ptr(why) }.to_string_lossy()))
}

// Err: why a slot callee on operands of these shapes is no entry: no line of
// LICENSE takes them, or what the slot would run for their classes
// (csrc/calls_slots.c) is not the C code the line speaks of.
pub unsafe fn license(callee: &Callee, operands: &[Shape]) -> Result<(), String> {
    unsafe {
        let Some((kind, op)) = kind(callee) else { return Err("no slot callee".into()) };
        let takes = |line: &Line| {
            line.kind == kind
                && (line.ops.is_empty() || line.ops.contains(&op.as_str()))
                && line.operands.len() == operands.len()
                && line.operands.iter().zip(operands).all(|(is, s)| is.holds(s))
        };
        if !LICENSE.iter().any(takes) {
            let classes = operands.iter().map(|s| class_text(s)).collect::<Vec<_>>().join(", ");
            return Err(format!("no line of the license table takes {kind}{}{op} on ({classes})", if op.is_empty() { "" } else { " " }));
        }
        let (none, name): (*mut PyTypeObject, *mut PyObject) = (std::ptr::null_mut(), std::ptr::null_mut());
        let at = |i: usize| class(&operands[i]);
        match callee {
            Callee::Binary(op) => second(0, *op as c_int, at(0), at(1), name)?,
            Callee::Compare(op) => second(1, *op as c_int, at(0), at(1), name)?,
            Callee::Unary(opcode, oparg) => second(2, UNARY_OPS.iter().position(|u| (u.1, u.2) == (*opcode, *oparg)).map_or(-1, |k| k as c_int), at(0), none, name)?,
            Callee::GetItem => second(3, 0, at(0), none, name)?,
            Callee::SetItem => second(4, 0, at(1), none, name)?,
            Callee::DelItem => second(5, 0, at(0), none, name)?,
            Callee::GetAttr(attr) => second(6, 0, at(0), none, attr.as_ptr())?,
            Callee::SetAttr(_) | Callee::Vector(..) | Callee::State(_) | Callee::Drop(_) => return Err("no slot callee with a line".into()),
        }
        // A tensor as a bound of a slice: its __index__.
        let key = match callee {
            Callee::GetItem => operands.get(1),
            Callee::SetItem => operands.get(2),
            Callee::Binary(_) | Callee::Compare(_) | Callee::Unary(..) | Callee::DelItem | Callee::GetAttr(_) | Callee::SetAttr(_) | Callee::Vector(..) | Callee::State(_) | Callee::Drop(_) => None,
        };
        let slices: Vec<&Shape> = match key {
            Some(Shape::Tuple(items)) => items.iter().collect(),
            Some(one) => vec![one],
            None => vec![],
        };
        for s in slices {
            if let Shape::Slice(bounds) = s {
                for b in bounds {
                    if Is::Tensor.holds(b) {
                        second(8, 0, class(b), none, name)?;
                    }
                }
            }
        }
        Ok(())
    }
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
