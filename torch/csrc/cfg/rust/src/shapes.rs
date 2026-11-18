//! The sizes a compiled region is called with.
//!
//! A block of the VM has versions, each with the guards it was specialized
//! under (`GuardCond`), and runs the first one whose guards hold. A region
//! that a graph backend compiled has variants in the same way, and what they
//! are guarded on is mostly sizes: of the tensors the region is handed, and
//! the integers it is handed next to them. Both are ordinary values of the
//! program, so one call's worth of them is a row of numbers (`tensor_key`),
//! and the condition under which a graph holds is an expression over that
//! row (`ShapeGuard`): every number is what it was when the graph was made,
//! except the ones the graph leaves open, which only have to satisfy what
//! its trace branched on.

use crate::capi::{self, TcfgTensorInfo};
use crate::pyffi;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyTuple};
use std::collections::HashMap;

type Obj = *mut ffi::PyObject;

/// What a list of tensors is like, for the key of a region's variants:
///
/// - their kinds: rank, element type, device, `requires_grad`, dispatch keys
///   and whether they have strides, as bytes that only compare equal for
///   tensors a graph can treat alike;
/// - their numbers: per tensor its sizes and then its strides, and after the
///   tensors the integers of `ints`, as native 64-bit integers;
/// - their storage offsets, one per tensor. Those are no part of what a
///   graph is keyed on, but a condition may mention them.
#[pyfunction]
pub fn tensor_key<'py>(
    py: Python<'py>,
    tensors: &Bound<'py, PyList>,
    ints: &Bound<'py, PyList>,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    let api = capi::api()
        .ok_or_else(|| PyRuntimeError::new_err("torch.rewrite is not initialized"))?;
    let count = tensors.len();
    let mut kinds: Vec<u8> = Vec::with_capacity(count * 24);
    let mut numbers: Vec<i64> = Vec::with_capacity(count * 8 + ints.len());
    let mut offsets: Vec<i64> = Vec::with_capacity(count);
    for tensor in tensors.iter() {
        let mut info = TcfgTensorInfo::empty();
        if unsafe { (api.tensor_info)(tensor.as_ptr(), &mut info) } == 0 {
            return Err(PyTypeError::new_err("tensor_key takes plain tensors"));
        }
        let rank = info.ndim as usize;
        let strided = !info.strides.is_null();
        kinds.extend_from_slice(&info.ndim.to_ne_bytes());
        kinds.extend_from_slice(&info.dtype.to_ne_bytes());
        kinds.extend_from_slice(&info.device.to_ne_bytes());
        kinds.extend_from_slice(&(info.requires_grad | (strided as i32) << 1).to_ne_bytes());
        kinds.extend_from_slice(&info.key_set.to_ne_bytes());
        unsafe {
            numbers.extend_from_slice(std::slice::from_raw_parts(info.sizes, rank));
            if strided {
                numbers.extend_from_slice(std::slice::from_raw_parts(info.strides, rank));
            }
        }
        offsets.push(info.storage_offset);
    }
    for value in ints.iter() {
        numbers.push(value.extract::<i64>()?);
    }
    let bytes = |row: &[i64]| unsafe {
        std::slice::from_raw_parts(row.as_ptr() as *const u8, std::mem::size_of_val(row))
    };
    Ok((
        PyBytes::new(py, &kinds),
        PyBytes::new(py, bytes(&numbers)),
        PyBytes::new(py, bytes(&offsets)),
    ))
}

/// A number that changes when one of the tensors is another object, has its
/// data somewhere else, or was written to in place: what a graph that has
/// the contents of the tensors built into it is good for. None if one of
/// them does not count its writes.
#[pyfunction]
pub fn tensor_stamp(tensors: &Bound<'_, PyList>) -> PyResult<Option<u64>> {
    let api = capi::api()
        .ok_or_else(|| PyRuntimeError::new_err("torch.rewrite is not initialized"))?;
    let mut stamp: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |v: u64| stamp = (stamp ^ v).wrapping_mul(0x0000_0100_0000_01b3);
    for tensor in tensors.iter() {
        let mut info = TcfgTensorInfo::empty();
        if unsafe { (api.tensor_info)(tensor.as_ptr(), &mut info) } == 0 {
            return Err(PyTypeError::new_err("tensor_stamp takes plain tensors"));
        }
        if info.version == capi::TCFG_NO_VERSION {
            return Ok(None);
        }
        mix(tensor.as_ptr() as u64);
        mix(info.data as u64);
        mix(info.version);
    }
    Ok(Some(stamp))
}

/// What the tensors of a module were when a region last looked at them one
/// by one: which object each place held, and of each tensor what the key of
/// a variant is made of and what a graph that has it built in depends on. A
/// call for which `holds` says that none of it has changed is handed what
/// the look found, without looking again.
#[pyclass(module = "torch._cfg_rs", name = "Snapshot", frozen)]
pub struct Snapshot {
    /// Per place (a dict and a key, or a list and an index): which of the
    /// tensors it held.
    places: Vec<u32>,
    tensors: Vec<Seen>,
    /// Which of the tensors an object is.
    index: HashMap<usize, u32>,
}

struct Seen {
    object: usize,
    data: usize,
    version: u64,
    /// Rank, element type, device, flags and dispatch keys, then the storage
    /// offset, the sizes and the strides.
    key: Vec<i64>,
}

unsafe fn seen(api: &capi::TcfgApi, tensor: *mut pyo3::ffi::PyObject) -> Option<Seen> {
    let mut info = TcfgTensorInfo::empty();
    if (api.tensor_info)(tensor, &mut info) == 0 {
        return None;
    }
    let rank = info.ndim as usize;
    let strided = !info.strides.is_null();
    let mut key = Vec::with_capacity(6 + 2 * rank);
    key.push(info.ndim as i64);
    key.push(info.dtype as i64);
    key.push(info.device as i64);
    key.push((info.requires_grad | (strided as i32) << 1) as i64);
    key.push(info.key_set as i64);
    key.push(info.storage_offset);
    key.extend_from_slice(std::slice::from_raw_parts(info.sizes, rank));
    if strided {
        key.extend_from_slice(std::slice::from_raw_parts(info.strides, rank));
    }
    Some(Seen { object: tensor as usize, data: info.data as usize, version: info.version, key })
}

/// What is at a place, borrowed; null if nothing is.
unsafe fn at_place(place: *mut pyo3::ffi::PyObject) -> *mut pyo3::ffi::PyObject {
    use pyo3::ffi;
    if ffi::PyTuple_Check(place) == 0 || ffi::PyTuple_GET_SIZE(place) != 2 {
        return std::ptr::null_mut();
    }
    let (store, name) = (ffi::PyTuple_GET_ITEM(place, 0), ffi::PyTuple_GET_ITEM(place, 1));
    if ffi::PyDict_CheckExact(store) != 0 {
        let found = ffi::PyDict_GetItemWithError(store, name);
        if found.is_null() {
            ffi::PyErr_Clear();
        }
        found
    } else if ffi::PyList_CheckExact(store) != 0 && ffi::PyLong_Check(name) != 0 {
        let index = ffi::PyLong_AsSsize_t(name);
        if index < 0 || index >= ffi::PyList_GET_SIZE(store) {
            ffi::PyErr_Clear();
            return std::ptr::null_mut();
        }
        ffi::PyList_GET_ITEM(store, index)
    } else {
        std::ptr::null_mut()
    }
}

/// The snapshot of the tensors at `places`, of which `tensors` are the
/// distinct ones in the order the places have them; None if something at a
/// place is not one of them or not a plain tensor.
#[pyfunction]
pub fn snapshot(places: &Bound<'_, PyList>, tensors: &Bound<'_, PyList>) -> PyResult<Option<Snapshot>> {
    let api = capi::api()
        .ok_or_else(|| PyRuntimeError::new_err("torch.rewrite is not initialized"))?;
    let mut found = Vec::with_capacity(tensors.len());
    for tensor in tensors.iter() {
        match unsafe { seen(api, tensor.as_ptr()) } {
            Some(one) => found.push(one),
            None => return Ok(None),
        }
    }
    let mut which = Vec::with_capacity(places.len());
    for place in places.iter() {
        let object = unsafe { at_place(place.as_ptr()) } as usize;
        match found.iter().position(|one| one.object == object) {
            Some(index) => which.push(index as u32),
            None => return Ok(None),
        }
    }
    let index = found.iter().enumerate().map(|(i, one)| (one.object, i as u32)).collect();
    Ok(Some(Snapshot { places: which, tensors: found, index }))
}

impl Snapshot {
    /// What `holds` says of the tensors themselves, for a caller that knows
    /// every place to hold the tensor it held.
    unsafe fn compare(&self, api: &capi::TcfgApi) -> u8 {
        let mut same = 2;
        for known in &self.tensors {
            let mut info = TcfgTensorInfo::empty();
            if (api.tensor_info)(known.object as Obj, &mut info) == 0 {
                return 0;
            }
            let rank = info.ndim as usize;
            let strided = !info.strides.is_null();
            let key = &known.key;
            let head = [
                info.ndim as i64,
                info.dtype as i64,
                info.device as i64,
                (info.requires_grad | (strided as i32) << 1) as i64,
                info.key_set as i64,
                info.storage_offset,
            ];
            if key.len() != 6 + rank * (1 + strided as usize)
                || key[..6] != head
                || key[6..6 + rank] != *std::slice::from_raw_parts(info.sizes, rank)
                || (strided && key[6 + rank..] != *std::slice::from_raw_parts(info.strides, rank))
            {
                return 0;
            }
            if info.data as usize != known.data
                || info.version != known.version
                || info.version == capi::TCFG_NO_VERSION
            {
                same = 1;
            }
        }
        same
    }
}

#[pymethods]
impl Snapshot {
    /// 0 if a place holds another object than it did, or a tensor is of
    /// another kind, size or layout; 1 if only what tensors hold may have
    /// changed (one was written to in place, or has its data elsewhere);
    /// 2 if nothing has.
    fn holds(&self, places: &Bound<'_, PyList>) -> u8 {
        let Some(api) = capi::api() else {
            return 0;
        };
        if places.len() != self.places.len() {
            return 0;
        }
        unsafe {
            for (i, which) in self.places.iter().enumerate() {
                let place = ffi::PyList_GET_ITEM(places.as_ptr(), i as isize);
                if at_place(place) as usize != self.tensors[*which as usize].object {
                    return 0;
                }
            }
            self.compare(api)
        }
    }

    /// `holds` without a look at the places: for a caller that knows that
    /// nothing the places are in has changed since `holds` was not 0 (see
    /// `module_epoch`).
    fn holds_in_place(&self) -> u8 {
        match capi::api() {
            Some(api) => unsafe { self.compare(api) },
            None => 0,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Binary {
    Add,
    Sub,
    Mul,
    FloorDiv,
    Mod,
    Min,
    Max,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// An integer computed from the numbers of a call. A truth value is 0 or 1.
enum Expr {
    Const(i64),
    /// One of the numbers of `tensor_key`, or, past the end of those, one of
    /// its storage offsets.
    Slot(usize),
    Neg(Box<Expr>),
    Not(Box<Expr>),
    Binary(Binary, Box<Expr>, Box<Expr>),
    All(Vec<Expr>),
    Any(Vec<Expr>),
}

fn number(bytes: &[u8], index: usize) -> i64 {
    let at = index * 8;
    i64::from_ne_bytes(bytes[at..at + 8].try_into().unwrap())
}

impl Expr {
    /// From what Python hands over: `("const", n)`, `("slot", i)`,
    /// `("neg" | "not", a)`, `("and" | "or", [a, ...])`, or the name of a
    /// binary operation with its two operands.
    fn parse(obj: &Bound<'_, PyAny>, slots: usize) -> PyResult<Expr> {
        let parts = obj.cast::<PyTuple>()?;
        let bad = || PyValueError::new_err(format!("not a size expression: {}", obj));
        if parts.len() < 2 {
            return Err(bad());
        }
        let name: String = parts.get_item(0)?.extract()?;
        let operand = |i: usize| -> PyResult<Box<Expr>> {
            Ok(Box::new(Expr::parse(&parts.get_item(i)?, slots)?))
        };
        let several = || -> PyResult<Vec<Expr>> {
            parts.get_item(1)?.try_iter()?.map(|part| Expr::parse(&part?, slots)).collect()
        };
        let binary = match name.as_str() {
            "const" => return Ok(Expr::Const(parts.get_item(1)?.extract()?)),
            "slot" => {
                let index: usize = parts.get_item(1)?.extract()?;
                if index >= slots {
                    return Err(bad());
                }
                return Ok(Expr::Slot(index));
            }
            "neg" => return Ok(Expr::Neg(operand(1)?)),
            "not" => return Ok(Expr::Not(operand(1)?)),
            "and" => return Ok(Expr::All(several()?)),
            "or" => return Ok(Expr::Any(several()?)),
            "add" => Binary::Add,
            "sub" => Binary::Sub,
            "mul" => Binary::Mul,
            "floordiv" => Binary::FloorDiv,
            "mod" => Binary::Mod,
            "min" => Binary::Min,
            "max" => Binary::Max,
            "eq" => Binary::Eq,
            "ne" => Binary::Ne,
            "lt" => Binary::Lt,
            "le" => Binary::Le,
            "gt" => Binary::Gt,
            "ge" => Binary::Ge,
            _ => return Err(bad()),
        };
        if parts.len() != 3 {
            return Err(bad());
        }
        Ok(Expr::Binary(binary, operand(1)?, operand(2)?))
    }

    /// None where Python would raise or go beyond 64 bits: a condition that
    /// cannot be computed does not hold.
    fn eval(&self, numbers: &[u8], offsets: &[u8]) -> Option<i64> {
        Some(match self {
            Expr::Const(value) => *value,
            Expr::Slot(index) => {
                let own = numbers.len() / 8;
                if *index < own {
                    number(numbers, *index)
                } else {
                    number(offsets, *index - own)
                }
            }
            Expr::Neg(a) => a.eval(numbers, offsets)?.checked_neg()?,
            Expr::Not(a) => (a.eval(numbers, offsets)? == 0) as i64,
            Expr::All(parts) => {
                for part in parts {
                    if part.eval(numbers, offsets)? == 0 {
                        return Some(0);
                    }
                }
                1
            }
            Expr::Any(parts) => {
                for part in parts {
                    if part.eval(numbers, offsets)? != 0 {
                        return Some(1);
                    }
                }
                0
            }
            Expr::Binary(op, a, b) => {
                let (a, b) = (a.eval(numbers, offsets)?, b.eval(numbers, offsets)?);
                match op {
                    Binary::Add => a.checked_add(b)?,
                    Binary::Sub => a.checked_sub(b)?,
                    Binary::Mul => a.checked_mul(b)?,
                    // As Python rounds: towards minus infinity.
                    Binary::FloorDiv => {
                        let q = a.checked_div(b)?;
                        if a % b != 0 && (a < 0) != (b < 0) {
                            q - 1
                        } else {
                            q
                        }
                    }
                    Binary::Mod => {
                        let r = a.checked_rem(b)?;
                        if r != 0 && (r < 0) != (b < 0) {
                            r + b
                        } else {
                            r
                        }
                    }
                    Binary::Min => a.min(b),
                    Binary::Max => a.max(b),
                    Binary::Eq => (a == b) as i64,
                    Binary::Ne => (a != b) as i64,
                    Binary::Lt => (a < b) as i64,
                    Binary::Le => (a <= b) as i64,
                    Binary::Gt => (a > b) as i64,
                    Binary::Ge => (a >= b) as i64,
                }
            }
        })
    }
}

/// When a graph with open sizes holds for a call.
#[pyclass(module = "torch._cfg_rs", frozen)]
pub struct ShapeGuard {
    /// The numbers the graph was made on. A call must have the same ones
    /// wherever `open` does not say otherwise.
    made_on: Vec<u8>,
    open: Vec<bool>,
    offsets: usize,
    conditions: Vec<Expr>,
}

#[pymethods]
impl ShapeGuard {
    /// `numbers` and `offsets` are those of `tensor_key` for the call the
    /// graph was made on, `open` the positions in `numbers` that the graph
    /// leaves open, `conditions` what must be true of a call for the graph to
    /// hold, over the numbers and, counted on from them, the offsets.
    #[new]
    fn new(
        numbers: &Bound<'_, PyBytes>,
        offsets: &Bound<'_, PyBytes>,
        open: Vec<usize>,
        conditions: &Bound<'_, PyList>,
    ) -> PyResult<ShapeGuard> {
        let made_on = numbers.as_bytes().to_vec();
        let count = made_on.len() / 8;
        let offsets = offsets.as_bytes().len() / 8;
        let mut is_open = vec![false; count];
        for index in open {
            match is_open.get_mut(index) {
                Some(slot) => *slot = true,
                None => return Err(PyValueError::new_err("no such number of the call")),
            }
        }
        let conditions = conditions
            .iter()
            .map(|c| Expr::parse(&c, count + offsets))
            .collect::<PyResult<Vec<Expr>>>()?;
        Ok(ShapeGuard { made_on, open: is_open, offsets, conditions })
    }

    /// Whether the graph holds for a call with these numbers and offsets.
    fn holds(&self, numbers: &Bound<'_, PyBytes>, offsets: &Bound<'_, PyBytes>) -> bool {
        let (numbers, offsets) = (numbers.as_bytes(), offsets.as_bytes());
        if numbers.len() != self.made_on.len() || offsets.len() != self.offsets * 8 {
            return false;
        }
        for (index, open) in self.open.iter().enumerate() {
            let at = index * 8;
            if !open && numbers[at..at + 8] != self.made_on[at..at + 8] {
                return false;
            }
        }
        self.conditions.iter().all(|c| matches!(c.eval(numbers, offsets), Some(v) if v != 0))
    }

    /// How many of the numbers are left open.
    #[getter]
    fn open(&self) -> usize {
        self.open.iter().filter(|open| **open).count()
    }
}

/// One thing a region is handed, as `Usual` expects it.
enum Node {
    /// A plain tensor, and which tensor of the call it is.
    Tensor(usize),
    /// An integer that is one of the numbers of the call.
    Int,
    /// A plain value: of the same type and equal.
    Value(Py<PyAny>),
    /// What is told by which object it is: a function, a class, a module
    /// without tensors of its own or inside one that came before.
    Object(Py<PyAny>),
    Tuple(Vec<Node>),
    List(Vec<Node>),
    /// A dict: its keys in order, and what they hold.
    Dict(Vec<Py<PyAny>>, Vec<Node>),
    /// A variable that a nested function shares: what it holds.
    Cell(Box<Node>),
    /// A pytree node: its class, what flattens it, the text of the context
    /// it flattens to, and what it holds.
    Flat(Py<PyAny>, Py<PyAny>, String, Vec<Node>),
    Module(Box<Looked>),
    /// A bound method: its function, and what it is bound to.
    Method(Py<PyAny>, Box<Node>),
    /// An object that is its class and its attributes: the class, the names
    /// in order, and what they hold.
    Instance(Py<PyAny>, Vec<Py<PyAny>>, Vec<Node>),
    /// A list, a dict, a cell or such an object that the call had before:
    /// which of the ones it holds.
    Again(usize),
}

/// A module whose tensors were found as a look had left them.
struct Looked {
    module: Py<PyAny>,
    /// Its `_ModuleState` and the `_Known` that had: it may have another.
    state: Py<PyAny>,
    known: Py<PyAny>,
    snapshot: Py<Snapshot>,
    tensors: Py<PyList>,
    /// Which tensor of the call its first one is.
    first: usize,
    kinds: Vec<u8>,
    numbers: Vec<u8>,
    offsets: Vec<u8>,
}

/// What a call of a region was like, for the calls after it that are like
/// it: the same modules with nothing in them changed, tensors, numbers and
/// plain values where it had them, in tuples, lists and dicts built the same
/// way. `look` tells whether a call is one, in one pass and without the
/// interpreter, and gives what describing it the long way would give: its
/// key and what the region is handed. What it does not know (a set, a
/// method of a builtin, a pytree node met twice) the long way is for.
#[pyclass(module = "torch._cfg_rs", name = "Usual", frozen)]
pub struct Usual {
    positions: Vec<usize>,
    nodes: Vec<Node>,
    /// What says nothing changed in a module: `module_epoch` and the modes
    /// of the thread, both as they were, and no hook for all modules.
    epoch: u64,
    modes: u64,
    hooks: Vec<Py<PyAny>>,
    /// The types a tensor may have, and how far an integer may go.
    types: Vec<usize>,
    wide: i64,
    /// Of the key: its parts and its modes; and what the modules were found
    /// to hold (`_Inputs.known`).
    parts: Py<PyAny>,
    key_modes: Py<PyAny>,
    known: Py<PyAny>,
    /// The key of the last call: the next one's is most often the same,
    /// and this one has its hash.
    last: std::sync::Mutex<Option<Py<PyTuple>>>,
}

fn node(spec: &Bound<'_, PyAny>) -> PyResult<Node> {
    let spec = spec.cast::<PyTuple>()?;
    let kind: String = spec.get_item(0)?.extract()?;
    let nodes = |items: Bound<'_, PyAny>| -> PyResult<Vec<Node>> {
        items.try_iter()?.map(|item| node(&item?)).collect()
    };
    let keys = |items: Bound<'_, PyAny>| -> PyResult<Vec<Py<PyAny>>> {
        items.try_iter()?.map(|key| Ok(key?.unbind())).collect()
    };
    Ok(match kind.as_str() {
        "tensor" => Node::Tensor(spec.get_item(1)?.extract()?),
        "int" => Node::Int,
        "value" => Node::Value(spec.get_item(1)?.unbind()),
        "object" => Node::Object(spec.get_item(1)?.unbind()),
        "tuple" => Node::Tuple(nodes(spec.get_item(1)?)?),
        "list" => Node::List(nodes(spec.get_item(1)?)?),
        "cell" => Node::Cell(Box::new(node(&spec.get_item(1)?)?)),
        "dict" => Node::Dict(keys(spec.get_item(1)?)?, nodes(spec.get_item(2)?)?),
        "method" => Node::Method(spec.get_item(1)?.unbind(), Box::new(node(&spec.get_item(2)?)?)),
        "instance" => Node::Instance(spec.get_item(1)?.unbind(), keys(spec.get_item(2)?)?, nodes(spec.get_item(3)?)?),
        "again" => Node::Again(spec.get_item(1)?.extract()?),
        "node" => Node::Flat(
            spec.get_item(1)?.unbind(),
            spec.get_item(2)?.unbind(),
            spec.get_item(3)?.extract()?,
            nodes(spec.get_item(4)?)?,
        ),
        "module" => {
            let known = spec.get_item(3)?;
            let bytes = |name: &str| -> PyResult<Vec<u8>> {
                Ok(known.getattr(name)?.cast::<PyBytes>()?.as_bytes().to_vec())
            };
            Node::Module(Box::new(Looked {
                module: spec.get_item(1)?.unbind(),
                state: spec.get_item(2)?.unbind(),
                snapshot: known.getattr("snapshot")?.cast::<Snapshot>()?.clone().unbind(),
                tensors: known.getattr("tensors")?.cast::<PyList>()?.clone().unbind(),
                first: spec.get_item(4)?.extract()?,
                kinds: bytes("kinds")?,
                numbers: bytes("numbers")?,
                offsets: bytes("offsets")?,
                known: known.unbind(),
            }))
        }
        other => return Err(PyValueError::new_err(format!("unknown kind {other:?}"))),
    })
}

/// What a walk over a call has found so far.
struct Walk<'a> {
    api: &'a capi::TcfgApi,
    usual: &'a Usual,
    /// The tensors of the call; of those, the ones that are no module's, with
    /// where they are; and the modules, whose tensors are at `first`.
    flat: Vec<Obj>,
    free: Vec<(Obj, usize)>,
    looked: Vec<&'a Looked>,
    /// The lists and dicts, and the integers with what they are.
    held: Vec<Obj>,
    ints: Vec<(Obj, i64)>,
    /// The pytree nodes with what they were flattened to, and the cells
    /// with what they hold: references kept for as long as the walk has
    /// what is in them.
    flattened: Vec<(Obj, Obj)>,
    unwritten: bool,
}

impl Drop for Walk<'_> {
    fn drop(&mut self) {
        for (_, parts) in &self.flattened {
            unsafe { ffi::Py_DECREF(*parts) };
        }
    }
}

impl<'a> Walk<'a> {
    fn kept(&self, obj: Obj) -> bool {
        self.flattened.iter().any(|(o, _)| *o == obj)
    }

    /// Whether a dict has these keys, in this order, and what is expected
    /// under them.
    unsafe fn items(&mut self, dict: Obj, keys: &'a [Py<PyAny>], values: &'a [Node]) -> bool {
        if ffi::PyDict_CheckExact(dict) == 0 || ffi::PyDict_Size(dict) as usize != keys.len() {
            return false;
        }
        let (mut at, mut key, mut value) = (0, std::ptr::null_mut(), std::ptr::null_mut());
        let mut items = Vec::with_capacity(keys.len());
        while ffi::PyDict_Next(dict, &mut at, &mut key, &mut value) != 0 {
            items.push((key, value));
        }
        items.len() == keys.len()
            && items.iter().zip(keys).all(|((key, _), expected)| same_value(*key, expected.as_ptr()))
            && items.iter().zip(values).all(|((_, value), node)| self.visit(node, *value))
    }

    unsafe fn visit(&mut self, node: &'a Node, obj: Obj) -> bool {
        match node {
            Node::Tensor(expected) => {
                if !self.usual.types.contains(&(ffi::Py_TYPE(obj) as usize)) {
                    return false;
                }
                let mut position = self.free.iter().find(|(o, _)| *o == obj).map(|(_, p)| *p);
                if position.is_none() {
                    position = self.looked.iter().find_map(|looked| {
                        let index = looked.snapshot.get().index.get(&(obj as usize))?;
                        Some(looked.first + *index as usize)
                    });
                }
                let position = position.unwrap_or_else(|| {
                    self.flat.push(obj);
                    self.free.push((obj, self.flat.len() - 1));
                    self.flat.len() - 1
                });
                position == *expected
            }
            Node::Int => {
                if ffi::PyLong_CheckExact(obj) == 0 {
                    return false;
                }
                let mut overflow = 0;
                let value = ffi::PyLong_AsLongLongAndOverflow(obj, &mut overflow);
                if overflow != 0 || value <= -self.usual.wide || value >= self.usual.wide {
                    return false;
                }
                self.ints.push((obj, value));
                true
            }
            Node::Value(expected) => same_value(obj, expected.as_ptr()),
            Node::Object(expected) => obj == expected.as_ptr(),
            Node::Tuple(items) => {
                ffi::PyTuple_CheckExact(obj) != 0
                    && ffi::PyTuple_GET_SIZE(obj) as usize == items.len()
                    && items.iter().enumerate().all(|(i, item)| self.visit(item, ffi::PyTuple_GET_ITEM(obj, i as isize)))
            }
            Node::List(items) => {
                if ffi::PyList_CheckExact(obj) == 0
                    || ffi::PyList_GET_SIZE(obj) as usize != items.len()
                    || self.held.contains(&obj)
                {
                    return false;
                }
                self.held.push(obj);
                // A list can change under the walk only through code the
                // walk runs, and it runs none but `==` of plain values.
                items.iter().enumerate().all(|(i, item)| {
                    i < ffi::PyList_GET_SIZE(obj) as usize && self.visit(item, ffi::PyList_GET_ITEM(obj, i as isize))
                })
            }
            Node::Dict(keys, values) => {
                if ffi::PyDict_CheckExact(obj) == 0 || self.held.contains(&obj) {
                    return false;
                }
                self.held.push(obj);
                self.items(obj, keys, values)
            }
            Node::Method(function, receiver) => {
                ffi::Py_TYPE(obj) == &raw mut pyffi::PyMethod_Type
                    && pyffi::PyMethod_Function(obj) == function.as_ptr()
                    && self.visit(receiver, pyffi::PyMethod_Self(obj))
            }
            Node::Instance(class, names, values) => {
                if ffi::Py_TYPE(obj) as Obj != class.as_ptr() || self.held.contains(&obj) {
                    return false;
                }
                self.held.push(obj);
                // A new reference to its `__dict__`, kept until the walk is
                // over.
                let state = ffi::PyObject_GenericGetDict(obj, std::ptr::null_mut());
                if state.is_null() {
                    ffi::PyErr_Clear();
                    return false;
                }
                self.flattened.push((obj, state));
                self.items(state, names, values)
            }
            Node::Again(index) => self.held.get(*index) == Some(&obj),
            Node::Cell(inner) => {
                if ffi::Py_TYPE(obj) != &raw mut pyffi::PyCell_Type || self.held.contains(&obj) {
                    return false;
                }
                self.held.push(obj);
                // A new reference, kept until the walk is over: the list of
                // what is held takes one of its own.
                let contents = pyffi::PyCell_Get(obj);
                if contents.is_null() {
                    return false;
                }
                self.flattened.push((obj, contents));
                self.visit(inner, contents)
            }
            Node::Flat(class, flatten, context, children) => {
                // Twice in a call, it is met "again": not what this is for.
                if ffi::Py_TYPE(obj) as Obj != class.as_ptr() || self.kept(obj) {
                    return false;
                }
                let parts = ffi::PyObject_CallOneArg(flatten.as_ptr(), obj);
                if parts.is_null() {
                    ffi::PyErr_Clear();
                    return false;
                }
                self.flattened.push((obj, parts));
                if ffi::PyTuple_CheckExact(parts) == 0 || ffi::PyTuple_GET_SIZE(parts) != 2 {
                    return false;
                }
                let (items, now) = (ffi::PyTuple_GET_ITEM(parts, 0), ffi::PyTuple_GET_ITEM(parts, 1));
                let text = ffi::PyObject_Repr(now);
                if text.is_null() {
                    ffi::PyErr_Clear();
                    return false;
                }
                let same = pyffi::str_to_string(text) == *context;
                ffi::Py_DECREF(text);
                if !same {
                    return false;
                }
                let count = if ffi::PyList_CheckExact(items) != 0 {
                    ffi::PyList_GET_SIZE(items)
                } else if ffi::PyTuple_CheckExact(items) != 0 {
                    ffi::PyTuple_GET_SIZE(items)
                } else {
                    return false;
                };
                count as usize == children.len()
                    && children.iter().enumerate().all(|(i, child)| {
                        let item = if ffi::PyList_CheckExact(items) != 0 {
                            ffi::PyList_GET_ITEM(items, i as isize)
                        } else {
                            ffi::PyTuple_GET_ITEM(items, i as isize)
                        };
                        self.visit(child, item)
                    })
            }
            Node::Module(looked) => {
                if obj != looked.module.as_ptr() {
                    return false;
                }
                static KNOWN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
                let name = *KNOWN.get_or_init(|| ffi::PyUnicode_InternFromString(c"known".as_ptr()) as usize);
                let known = ffi::PyObject_GetAttr(looked.state.as_ptr(), name as Obj);
                if known.is_null() {
                    ffi::PyErr_Clear();
                    return false;
                }
                ffi::Py_DECREF(known);
                let snapshot = looked.snapshot.get();
                // The look it had is the one that was taken, none of its
                // tensors came on their own before it, and they are what
                // they were.
                if known != looked.known.as_ptr()
                    || self.flat.len() != looked.first
                    || self.free.iter().any(|(o, _)| snapshot.index.contains_key(&(*o as usize)))
                {
                    return false;
                }
                let same = snapshot.compare(self.api);
                if same == 0 {
                    return false;
                }
                self.unwritten = self.unwritten && same == 2;
                let tensors = looked.tensors.as_ptr();
                for i in 0..ffi::PyList_GET_SIZE(tensors) {
                    self.flat.push(ffi::PyList_GET_ITEM(tensors, i));
                }
                self.looked.push(looked);
                true
            }
        }
    }
}

/// Whether two plain values are of one type and equal.
unsafe fn same_value(a: Obj, b: Obj) -> bool {
    if a == b {
        return true;
    }
    if ffi::Py_TYPE(a) != ffi::Py_TYPE(b) {
        return false;
    }
    let equal = ffi::PyObject_RichCompareBool(a, b, ffi::Py_EQ);
    if equal < 0 {
        ffi::PyErr_Clear();
    }
    equal == 1
}

unsafe fn list_of(py: Python<'_>, items: impl ExactSizeIterator<Item = Obj>) -> PyResult<Bound<'_, PyAny>> {
    let list = ffi::PyList_New(items.len() as isize);
    if list.is_null() {
        return Err(PyErr::fetch(py));
    }
    for (i, item) in items.enumerate() {
        ffi::Py_INCREF(item);
        ffi::PyList_SET_ITEM(list, i as isize, item);
    }
    Ok(Bound::from_owned_ptr(py, list))
}

#[pymethods]
impl Usual {
    /// `positions`: which of the values a region is handed its key is made
    /// of. `spec`: per position, what was there, as a tuple that begins
    /// with `"tensor"` (and which one of the call), `"int"`, `"value"` or
    /// `"object"` (and the object), `"tuple"` or `"list"` (and the items),
    /// `"cell"` (what it holds), `"dict"` (the keys, the values), `"method"` (the function of a bound
    /// method, what it is bound to), `"instance"` (the class of an object that is its attributes,
    /// their names, their values), `"again"` (which of the lists, dicts, cells and objects the call
    /// has met it is), `"node"` (the class of a pytree
    /// node, what flattens it, the text of its context, its children) or
    /// `"module"` (the module, its state, what its tensors were found to
    /// be, the first one's place in the call). `parts` and `modes` are those of the key, `known` is
    /// `_Inputs.known`, `types` the types of a tensor, `wide` the limit of
    /// an integer that counts as a number, `hooks` the dicts that must stay
    /// empty.
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        positions: Vec<usize>,
        spec: &Bound<'_, PyList>,
        parts: Py<PyAny>,
        modes: Py<PyAny>,
        known: Py<PyAny>,
        types: &Bound<'_, PyTuple>,
        wide: i64,
        hooks: &Bound<'_, PyTuple>,
    ) -> PyResult<Self> {
        let api = capi::api().ok_or_else(|| PyRuntimeError::new_err("torch.rewrite is not initialized"))?;
        if positions.len() != spec.len() {
            return Err(PyValueError::new_err("one spec per position"));
        }
        Ok(Usual {
            positions,
            nodes: spec.iter().map(|one| node(&one)).collect::<PyResult<_>>()?,
            epoch: pyffi::modules_version(),
            modes: unsafe { (api.call_modes)() },
            hooks: hooks.iter().map(|d| d.unbind()).collect(),
            types: types.iter().map(|t| t.as_ptr() as usize).collect(),
            wide,
            parts,
            key_modes: modes,
            known,
            last: std::sync::Mutex::new(None),
        })
    }

    /// For a call that is like the one this was made from: its key, the
    /// tensors, the lists and dicts, the integers, the storage offsets of
    /// the tensors, what the modules were found to hold and whether none of
    /// their tensors was written to; as `_describe` and `_key` find them.
    /// None for any other call.
    fn look<'py>(&self, py: Python<'py>, args: &Bound<'py, PyTuple>) -> PyResult<Option<Bound<'py, PyTuple>>> {
        let Some(api) = capi::api() else {
            return Ok(None);
        };
        unsafe {
            if pyffi::modules_version() != self.epoch
                || (api.call_modes)() != self.modes
                || self.hooks.iter().any(|d| ffi::PyDict_Size(d.as_ptr()) != 0)
            {
                return Ok(None);
            }
            let count = ffi::PyTuple_GET_SIZE(args.as_ptr()) as usize;
            let mut walk = Walk {
                api,
                usual: self,
                flat: Vec::with_capacity(64),
                free: Vec::new(),
                looked: Vec::new(),
                held: Vec::new(),
                ints: Vec::new(),
                flattened: Vec::new(),
                unwritten: true,
            };
            for (position, node) in self.positions.iter().zip(&self.nodes) {
                if *position >= count || !walk.visit(node, ffi::PyTuple_GET_ITEM(args.as_ptr(), *position as isize)) {
                    return Ok(None);
                }
            }
            // The kinds, numbers and offsets of the tensors, as `tensor_key`
            // gives them; those of a module are what its look found.
            let total = walk.flat.len();
            let mut kinds: Vec<u8> = Vec::with_capacity(total * 24);
            let mut numbers: Vec<u8> = Vec::with_capacity(total * 64);
            let mut offsets: Vec<u8> = Vec::with_capacity(total * 8);
            let mut looked = walk.looked.iter().peekable();
            let mut at = 0;
            while at < total {
                if let Some(module) = looked.peek().filter(|module| module.first == at) {
                    kinds.extend_from_slice(&module.kinds);
                    numbers.extend_from_slice(&module.numbers);
                    offsets.extend_from_slice(&module.offsets);
                    at += ffi::PyList_GET_SIZE(module.tensors.as_ptr()) as usize;
                    looked.next();
                    continue;
                }
                let mut info = TcfgTensorInfo::empty();
                if (api.tensor_info)(walk.flat[at], &mut info) == 0 {
                    return Ok(None);
                }
                let rank = info.ndim as usize;
                let strided = !info.strides.is_null();
                kinds.extend_from_slice(&info.ndim.to_ne_bytes());
                kinds.extend_from_slice(&info.dtype.to_ne_bytes());
                kinds.extend_from_slice(&info.device.to_ne_bytes());
                kinds.extend_from_slice(&(info.requires_grad | (strided as i32) << 1).to_ne_bytes());
                kinds.extend_from_slice(&info.key_set.to_ne_bytes());
                let bytes = |row: &[i64]| std::slice::from_raw_parts(row.as_ptr() as *const u8, std::mem::size_of_val(row));
                numbers.extend_from_slice(bytes(std::slice::from_raw_parts(info.sizes, rank)));
                if strided {
                    numbers.extend_from_slice(bytes(std::slice::from_raw_parts(info.strides, rank)));
                }
                offsets.extend_from_slice(&info.storage_offset.to_ne_bytes());
                at += 1;
            }
            for (_, value) in &walk.ints {
                numbers.extend_from_slice(&value.to_ne_bytes());
            }
            let mut last = self.last.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            let same = last.as_ref().filter(|key| {
                let bytes = |at: isize| {
                    let part = ffi::PyTuple_GET_ITEM(key.as_ptr(), at);
                    let (start, size) = (ffi::PyBytes_AsString(part), ffi::PyBytes_Size(part));
                    std::slice::from_raw_parts(start as *const u8, size as usize)
                };
                bytes(1) == kinds.as_slice() && bytes(2) == numbers.as_slice()
            });
            let key = match same {
                Some(key) => key.bind(py).clone(),
                None => {
                    let key = PyTuple::new(
                        py,
                        [
                            self.parts.bind(py).clone(),
                            PyBytes::new(py, &kinds).into_any(),
                            PyBytes::new(py, &numbers).into_any(),
                            self.key_modes.bind(py).clone(),
                        ],
                    )?;
                    *last = Some(key.clone().unbind());
                    key
                }
            };
            drop(last);
            let found = PyTuple::new(
                py,
                [
                    key.into_any(),
                    list_of(py, walk.flat.iter().copied())?,
                    list_of(py, walk.held.iter().copied())?,
                    list_of(py, walk.ints.iter().map(|(obj, _)| *obj))?,
                    PyBytes::new(py, &offsets).into_any(),
                    self.known.bind(py).clone(),
                    pyo3::types::PyBool::new(py, walk.unwritten).to_owned().into_any(),
                ],
            )?;
            Ok(Some(found))
        }
    }
}
