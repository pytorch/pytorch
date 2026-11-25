// A type lattice over the values seen at each SSA value (instr results and
// block params) while executing a Cfg. Each observed object is abstracted to a
// Ty and joined into the value's Ty so far. Observing never runs Python code:
// only exact builtin types are looked into (a subclass may override anything),
// and tensors are read through tensor_meta.
//
// Observing costs a bounded amount per value: containers are recorded by kind
// and length only (small tuples per position, one level deep). What a
// container holds shows up where it is used, as the types of the values read
// out of it (subscripts, iteration, unpacking), the way Dynamo only learns
// about the parts of an input that the code touches.

use std::fmt;

use pyo3::ffi::{self, PyObject};
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::IntoPyObjectExt;

use crate::bindings::ffi::{device_name, dtype_name, tensor_by_type, tensor_meta, TensorMeta};

// More alternatives than this in a union widen it to Any.
const MAX_UNION: usize = 4;
// Strings longer than this are tracked as str without their value.
const MAX_STR: usize = 64;
// Longer tuples, and tuples inside a tuple, are recorded by length only.
const MAX_TUPLE: usize = 16;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Interval {
    pub lo: i64,
    pub hi: i64,
}

impl Interval {
    pub fn exact(n: i64) -> Self {
        Interval { lo: n, hi: n }
    }

    fn join(self, o: Interval) -> Self {
        Interval {
            lo: self.lo.min(o.lo),
            hi: self.hi.max(o.hi),
        }
    }
}

impl fmt::Display for Interval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.lo, self.hi) {
            (lo, hi) if lo == hi => write!(f, "{lo}"),
            (i64::MIN, i64::MAX) => write!(f, "?"),
            (lo, hi) => write!(f, "{lo}..{hi}"),
        }
    }
}

// A value that is either always the same or not.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Konst<T> {
    Is(T),
    Varies,
}

impl<T: PartialEq> Konst<T> {
    fn absorb(&mut self, v: T) {
        if matches!(self, Konst::Is(a) if *a != v) {
            *self = Konst::Varies;
        }
    }
}

// An exact Python class, by identity. Not a reference: a class freed and
// another allocated at the same address would be conflated.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Class {
    ptr: usize,
    // tp_name, which for a heap type points into its name str.
    name_ptr: usize,
    name: String,
}

impl Class {
    unsafe fn of(obj: *mut PyObject) -> Self {
        let tp = ffi::Py_TYPE(obj);
        let name = std::ffi::CStr::from_ptr((*tp).tp_name).to_string_lossy().into_owned();
        Class { ptr: tp as usize, name_ptr: (*tp).tp_name as usize, name }
    }

    pub fn is(&self, tp: *const ffi::PyTypeObject) -> bool {
        self.ptr == tp as usize
    }

    // Whether obj's class is this one.
    unsafe fn of_obj(&self, obj: *mut PyObject) -> bool {
        let tp = ffi::Py_TYPE(obj);
        if self.ptr != tp as usize {
            return false;
        }
        if (*tp).tp_name as usize == self.name_ptr {
            return true;
        }
        // Byte by byte up to the NUL: names are short, and this is per object.
        let mut p = (*tp).tp_name as *const u8;
        for &b in self.name.as_bytes() {
            if *p != b {
                return std::ffi::CStr::from_ptr((*tp).tp_name).to_string_lossy() == self.name;
            }
            p = p.add(1);
        }
        *p == 0 || std::ffi::CStr::from_ptr((*tp).tp_name).to_string_lossy() == self.name
    }
}

#[derive(Clone, PartialEq, Debug)]
pub enum Shape {
    Dims(Vec<Interval>),
    // Not known when observed (Python-dispatch or symbolic sizes) or rank varies.
    Unknown,
}

#[derive(Clone, PartialEq, Debug)]
pub struct TensorTy {
    class: Class,
    // c10::ScalarType; Varies also when the tensor was undefined.
    dtype: Konst<i8>,
    // (c10::DeviceType, index).
    device: Konst<(i8, i8)>,
    requires_grad: Konst<bool>,
    shape: Shape,
}

impl TensorTy {
    pub fn class(&self) -> &Class {
        &self.class
    }

    unsafe fn absorb(&mut self, t: &crate::bindings::ffi::Tensor) {
        let mut m = TensorMeta::default();
        tensor_meta(t, &mut m);
        if !m.defined {
            (self.dtype, self.device, self.requires_grad, self.shape) = (Konst::Varies, Konst::Varies, Konst::Varies, Shape::Unknown);
            return;
        }
        self.dtype.absorb(m.dtype);
        if m.has_device {
            self.device.absorb((m.device_type, m.device_index));
        } else {
            self.device = Konst::Varies;
        }
        self.requires_grad.absorb(m.requires_grad);
        match &mut self.shape {
            Shape::Dims(d) if m.has_shape && d.len() == m.sizes.len() => {
                d.iter_mut().zip(&m.sizes).for_each(|(i, s)| *i = i.join(Interval::exact(*s)));
            }
            _ => self.shape = Shape::Unknown,
        }
    }

}

#[derive(Clone, PartialEq, Debug)]
pub enum Kind {
    // CPython's NULL (e.g. the self slot of a non-method LoadMethod).
    Null,
    None,
    Bool(Konst<bool>),
    // Beyond i64 joins to the full range.
    Int(Interval),
    Float,
    Str(Konst<String>),
    Tensor(TensorTy),
    // Per position while every tuple seen had the same length.
    Tuple(Vec<Ty>),
    // Lengths only.
    VarTuple(Interval),
    List(Interval),
    Set(Interval),
    FrozenSet(Interval),
    Dict(Interval),
    // Anything else, by exact class.
    Object(Class),
}

impl Kind {
    fn tuple_len(&self) -> Interval {
        match self {
            Kind::Tuple(elems) => Interval::exact(elems.len() as i64),
            Kind::VarTuple(len) => *len,
            _ => unreachable!("only tuples"),
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub enum Ty {
    // One alternative, inline: what most values have.
    One(Kind),
    // Two to MAX_UNION alternatives, no object absorbed by two.
    Union(Vec<Kind>),
    Any,
}

impl Kind {
    // Joins obj in if it's of this kind (for tensors and objects, this
    // class; tuples of any length); otherwise false.
    unsafe fn absorb(&mut self, obj: *mut PyObject, in_tuple: bool) -> bool {
        if obj.is_null() || matches!(self, Kind::Null) {
            return obj.is_null() && matches!(self, Kind::Null);
        }
        match self {
            Kind::Null => unreachable!(),
            Kind::None => obj == ffi::Py_None(),
            Kind::Bool(k) => {
                let is = ffi::PyBool_Check(obj) != 0;
                if is {
                    k.absorb(obj == ffi::Py_True());
                }
                is
            }
            Kind::Int(i) => {
                let is = ffi::PyLong_CheckExact(obj) != 0;
                if is {
                    *i = i.join(int_interval(obj));
                }
                is
            }
            Kind::Float => ffi::PyFloat_CheckExact(obj) != 0,
            Kind::Str(k) => {
                let is = ffi::PyUnicode_CheckExact(obj) != 0;
                if is && matches!(k, Konst::Is(s) if !str_equals(obj, s)) {
                    *k = Konst::Varies;
                }
                is
            }
            Kind::Tensor(t) => match tensor_by_type(obj as usize).as_ref() {
                Some(tt) if t.class.of_obj(obj) => {
                    t.absorb(tt);
                    true
                }
                _ => false,
            },
            Kind::Tuple(_) | Kind::VarTuple(_) => {
                if ffi::PyTuple_CheckExact(obj) == 0 {
                    return false;
                }
                let n = ffi::PyTuple_GET_SIZE(obj);
                match self {
                    Kind::Tuple(elems) if !in_tuple && n as usize <= MAX_TUPLE && n as usize == elems.len() => {
                        for (i, e) in elems.iter_mut().enumerate() {
                            e.absorb_in(ffi::PyTuple_GET_ITEM(obj, i as isize), true);
                        }
                    }
                    _ => *self = Kind::VarTuple(self.tuple_len().join(Interval::exact(n as i64))),
                }
                true
            }
            Kind::List(i) => absorb_len(i, ffi::PyList_CheckExact(obj) != 0, || ffi::PyList_GET_SIZE(obj)),
            Kind::Set(i) => absorb_len(i, ffi::PySet_CheckExact(obj) != 0, || ffi::PySet_Size(obj)),
            Kind::FrozenSet(i) => absorb_len(i, ffi::PyFrozenSet_CheckExact(obj) != 0, || ffi::PySet_Size(obj)),
            Kind::Dict(i) => absorb_len(i, ffi::PyDict_CheckExact(obj) != 0, || ffi::PyDict_Size(obj)),
            // Builtins and tensors never get Object, so the class decides.
            Kind::Object(c) => c.of_obj(obj),
        }
    }
}

fn absorb_len(i: &mut Interval, is: bool, len: impl FnOnce() -> isize) -> bool {
    if is {
        *i = i.join(Interval::exact(len() as i64));
    }
    is
}

impl Ty {
    // Joins obj's Ty in, allocating only when obj is of a kind not seen here
    // before.
    pub unsafe fn absorb(&mut self, obj: *mut PyObject) {
        self.absorb_in(obj, false)
    }

    // The alternatives, or None for Any.
    pub fn kinds(&self) -> Option<&[Kind]> {
        match self {
            Ty::One(k) => Some(std::slice::from_ref(k)),
            Ty::Union(alts) => Some(alts),
            Ty::Any => None,
        }
    }

    unsafe fn absorb_in(&mut self, obj: *mut PyObject, in_tuple: bool) {
        let alts = match self {
            Ty::Any => return,
            Ty::One(k) => {
                if !k.absorb(obj, in_tuple) {
                    let Ty::One(k) = std::mem::replace(self, Ty::Any) else { unreachable!() };
                    *self = Ty::Union(vec![k, kind(obj, in_tuple)]);
                }
                return;
            }
            Ty::Union(alts) => alts,
        };
        if alts.iter_mut().any(|a| a.absorb(obj, in_tuple)) {
            return;
        }
        alts.push(kind(obj, in_tuple));
        if alts.len() > MAX_UNION {
            *self = Ty::Any;
        }
    }

}

// A value's Ty so far, and objects that wouldn't change it: of class
// skip.0 with tp_name skip.1 (as Class::of_obj takes them to be the same), or
// any with SKIP_ALL. Checked before observing, which most observations then
// don't need.
#[derive(Clone, Default)]
pub struct Slot {
    pub ty: Option<Ty>,
    skip: (usize, usize),
}

const SKIP_ALL: usize = usize::MAX;

impl Slot {
    #[inline]
    pub unsafe fn skips(&self, obj: *mut PyObject) -> bool {
        if self.skip.0 == SKIP_ALL {
            return true;
        }
        if obj.is_null() {
            return false;
        }
        let tp = ffi::Py_TYPE(obj);
        tp as usize == self.skip.0 && (*tp).tp_name as usize == self.skip.1
    }

    // `constant`: the value is always this object, and it's immutable.
    pub unsafe fn observe(&mut self, obj: *mut PyObject, constant: bool) {
        let ty = match &mut self.ty {
            Some(t) => {
                t.absorb(obj);
                t
            }
            None => self.ty.insert(observe(obj)),
        };
        self.skip = if constant { (SKIP_ALL, 0) } else { ty.skip_key(obj) };
    }
}

impl Ty {
    // What Slot::skip can be after obj was absorbed: obj's class, if no
    // object of it would change the alternative obj went into.
    unsafe fn skip_key(&self, obj: *mut PyObject) -> (usize, usize) {
        let Some(kinds) = self.kinds() else {
            return (SKIP_ALL, 0);
        };
        if obj.is_null() {
            return (0, 0);
        }
        let tp = ffi::Py_TYPE(obj);
        let full = Interval { lo: i64::MIN, hi: i64::MAX };
        let saturated = kinds.iter().any(|k| match k {
            Kind::None => obj == ffi::Py_None(),
            Kind::Float => ffi::PyFloat_CheckExact(obj) != 0,
            Kind::Bool(Konst::Varies) => ffi::PyBool_Check(obj) != 0,
            Kind::Str(Konst::Varies) => ffi::PyUnicode_CheckExact(obj) != 0,
            Kind::Int(i) => *i == full && ffi::PyLong_CheckExact(obj) != 0,
            Kind::Object(c) => c.ptr == tp as usize && c.name_ptr == (*tp).tp_name as usize,
            _ => false,
        });
        if saturated {
            (tp as usize, (*tp).tp_name as usize)
        } else {
            (0, 0)
        }
    }
}

// The Ty of one object (borrowed, possibly NULL). Holds the GIL.
unsafe fn observe(obj: *mut PyObject) -> Ty {
    Ty::One(kind(obj, false))
}

unsafe fn kind(obj: *mut PyObject, in_tuple: bool) -> Kind {
    if obj.is_null() {
        return Kind::Null;
    }
    if obj == ffi::Py_None() {
        return Kind::None;
    }
    if ffi::PyBool_Check(obj) != 0 {
        return Kind::Bool(Konst::Is(obj == ffi::Py_True()));
    }
    if ffi::PyLong_CheckExact(obj) != 0 {
        return Kind::Int(int_interval(obj));
    }
    if ffi::PyFloat_CheckExact(obj) != 0 {
        return Kind::Float;
    }
    if ffi::PyUnicode_CheckExact(obj) != 0 {
        return Kind::Str(str_value(obj));
    }
    if let Some(t) = tensor_by_type(obj as usize).as_ref() {
        return Kind::Tensor(tensor_ty(obj, t));
    }
    if ffi::PyTuple_CheckExact(obj) != 0 {
        let n = ffi::PyTuple_GET_SIZE(obj);
        if in_tuple || n as usize > MAX_TUPLE {
            return Kind::VarTuple(Interval::exact(n as i64));
        }
        return Kind::Tuple((0..n).map(|i| Ty::One(kind(ffi::PyTuple_GET_ITEM(obj, i), true))).collect());
    }
    if ffi::PyList_CheckExact(obj) != 0 {
        return Kind::List(Interval::exact(ffi::PyList_GET_SIZE(obj) as i64));
    }
    if ffi::PySet_CheckExact(obj) != 0 {
        return Kind::Set(Interval::exact(ffi::PySet_Size(obj) as i64));
    }
    if ffi::PyFrozenSet_CheckExact(obj) != 0 {
        return Kind::FrozenSet(Interval::exact(ffi::PySet_Size(obj) as i64));
    }
    if ffi::PyDict_CheckExact(obj) != 0 {
        return Kind::Dict(Interval::exact(ffi::PyDict_Size(obj) as i64));
    }
    Kind::Object(Class::of(obj))
}

unsafe fn int_interval(obj: *mut PyObject) -> Interval {
    let mut overflow = 0;
    let n = ffi::PyLong_AsLongLongAndOverflow(obj, &mut overflow);
    match overflow {
        0 => Interval::exact(n),
        _ => Interval { lo: i64::MIN, hi: i64::MAX },
    }
}

// Whether str obj's value is s (by code point, as str_value reads it).
unsafe fn str_equals(obj: *mut PyObject, s: &str) -> bool {
    let n = ffi::PyUnicode_GET_LENGTH(obj);
    if ffi::PyUnicode_IS_ASCII(obj) != 0 {
        return n as usize == s.len() && std::slice::from_raw_parts(ffi::PyUnicode_1BYTE_DATA(obj), s.len()) == s.as_bytes();
    }
    let mut i = 0;
    for c in s.chars() {
        if i == n || ffi::PyUnicode_ReadChar(obj, i) != c as u32 {
            return false;
        }
        i += 1;
    }
    i == n
}

// Read by code point: PyUnicode_AsUTF8AndSize would cache a UTF-8 copy in a
// non-ASCII str, which sys.getsizeof can see.
unsafe fn str_value(obj: *mut PyObject) -> Konst<String> {
    let n = ffi::PyUnicode_GetLength(obj);
    if n as usize > MAX_STR {
        return Konst::Varies;
    }
    let mut out = String::new();
    for i in 0..n {
        // Lone surrogates aren't chars.
        match char::from_u32(ffi::PyUnicode_ReadChar(obj, i)) {
            Some(c) => out.push(c),
            None => return Konst::Varies,
        }
    }
    Konst::Is(out)
}

unsafe fn tensor_ty(obj: *mut PyObject, t: &crate::bindings::ffi::Tensor) -> TensorTy {
    let mut m = TensorMeta::default();
    tensor_meta(t, &mut m);
    let class = Class::of(obj);
    if !m.defined {
        return TensorTy {
            class,
            dtype: Konst::Varies,
            device: Konst::Varies,
            requires_grad: Konst::Varies,
            shape: Shape::Unknown,
        };
    }
    TensorTy {
        class,
        dtype: Konst::Is(m.dtype),
        device: if m.has_device {
            Konst::Is((m.device_type, m.device_index))
        } else {
            Konst::Varies
        },
        requires_grad: Konst::Is(m.requires_grad),
        shape: if m.has_shape {
            Shape::Dims(m.sizes.iter().map(|s| Interval::exact(*s)).collect())
        } else {
            Shape::Unknown
        },
    }
}

impl Ty {
    // As data for torch.vm's Type: None for Any, else a list of the
    // alternatives, each a tuple led by its kind's name.
    pub fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.kinds() {
            None => Ok(py.None().into_bound(py)),
            Some(kinds) => PyList::new(py, kinds.iter().map(|k| k.to_py(py)).collect::<PyResult<Vec<_>>>()?)?.into_bound_py_any(py),
        }
    }
}

impl Kind {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        fn konst<T: Clone>(k: &Konst<T>) -> Option<T> {
            match k {
                Konst::Is(v) => Some(v.clone()),
                Konst::Varies => None,
            }
        }
        let len = |name: &str, i: &Interval| (name, i.lo, i.hi).into_bound_py_any(py);
        match self {
            Kind::Null => ("null",).into_bound_py_any(py),
            Kind::None => ("none",).into_bound_py_any(py),
            Kind::Bool(b) => ("bool", konst(b)).into_bound_py_any(py),
            Kind::Int(i) => len("int", i),
            Kind::Float => ("float",).into_bound_py_any(py),
            Kind::Str(s) => ("str", konst(s)).into_bound_py_any(py),
            Kind::Tensor(t) => {
                let dtype = konst(&t.dtype).map(|d| dtype_name(d).to_string());
                let device = konst(&t.device).map(|(ty, i)| device_name(ty, i).to_string());
                let shape = match &t.shape {
                    Shape::Dims(d) => Some(d.iter().map(|i| (i.lo, i.hi)).collect::<Vec<_>>()),
                    Shape::Unknown => None,
                };
                ("tensor", t.class.name.as_str(), dtype, device, shape, konst(&t.requires_grad)).into_bound_py_any(py)
            }
            Kind::Tuple(elems) => ("tuple", PyList::new(py, elems.iter().map(|e| e.to_py(py)).collect::<PyResult<Vec<_>>>()?)?).into_bound_py_any(py),
            Kind::VarTuple(i) => len("tuple", i),
            Kind::List(i) => len("list", i),
            Kind::Set(i) => len("set", i),
            Kind::FrozenSet(i) => len("frozenset", i),
            Kind::Dict(i) => len("dict", i),
            Kind::Object(c) => ("object", c.name.as_str()).into_bound_py_any(py),
        }
    }
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Ty::Any => write!(f, "Any"),
            Ty::One(_) | Ty::Union(_) => {
                for (i, k) in self.kinds().into_iter().flatten().enumerate() {
                    if i > 0 {
                        write!(f, " | ")?;
                    }
                    write!(f, "{k}")?;
                }
                Ok(())
            }
        }
    }
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Kind::Null => write!(f, "null"),
            Kind::None => write!(f, "None"),
            Kind::Bool(Konst::Is(b)) => write!(f, "{}", if *b { "True" } else { "False" }),
            Kind::Bool(Konst::Varies) => write!(f, "bool"),
            Kind::Int(i) if i.lo == i.hi => write!(f, "int {i}"),
            Kind::Int(i) => write!(f, "int[{i}]"),
            Kind::Float => write!(f, "float"),
            Kind::Str(Konst::Is(s)) => write!(f, "str {s:?}"),
            Kind::Str(Konst::Varies) => write!(f, "str"),
            Kind::Tensor(t) => write!(f, "{t}"),
            Kind::Tuple(elems) => {
                let elems: Vec<String> = elems.iter().map(|e| e.to_string()).collect();
                write!(f, "tuple({})", elems.join(", "))
            }
            Kind::VarTuple(len) => write!(f, "tuple[{len}]"),
            Kind::List(len) => write!(f, "list[{len}]"),
            Kind::Set(len) => write!(f, "set[{len}]"),
            Kind::FrozenSet(len) => write!(f, "frozenset[{len}]"),
            Kind::Dict(len) => write!(f, "dict[{len}]"),
            Kind::Object(c) => write!(f, "{}", c.name),
        }
    }
}

impl fmt::Display for TensorTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = vec![];
        parts.push(match self.dtype {
            Konst::Is(d) => dtype_name(d).to_string(),
            Konst::Varies => "?".into(),
        });
        parts.push(match self.device {
            Konst::Is((t, i)) => device_name(t, i).to_string(),
            Konst::Varies => "?".into(),
        });
        parts.push(match &self.shape {
            Shape::Dims(d) => format!("[{}]", d.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(", ")),
            Shape::Unknown => "[...]".into(),
        });
        match self.requires_grad {
            Konst::Is(true) => parts.push("grad".into()),
            Konst::Is(false) => {}
            Konst::Varies => parts.push("grad?".into()),
        }
        write!(f, "{}({})", self.class.name, parts.join(" "))
    }
}
