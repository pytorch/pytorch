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
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyTuple};

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
        PyBytes::new_bound(py, &kinds),
        PyBytes::new_bound(py, bytes(&numbers)),
        PyBytes::new_bound(py, bytes(&offsets)),
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
    Ok(Some(Snapshot { places: which, tensors: found }))
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
        let mut checked = vec![false; self.tensors.len()];
        let mut same = 2;
        for (i, which) in self.places.iter().enumerate() {
            let known = &self.tensors[*which as usize];
            unsafe {
                let place = pyo3::ffi::PyList_GET_ITEM(places.as_ptr(), i as isize);
                let object = at_place(place);
                if object as usize != known.object {
                    return 0;
                }
                if std::mem::replace(&mut checked[*which as usize], true) {
                    continue;
                }
                let mut info = TcfgTensorInfo::empty();
                if (api.tensor_info)(object, &mut info) == 0 {
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
        }
        same
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
        let parts = obj.downcast::<PyTuple>()?;
        let bad = || PyValueError::new_err(format!("not a size expression: {}", obj));
        if parts.len() < 2 {
            return Err(bad());
        }
        let name: String = parts.get_item(0)?.extract()?;
        let operand = |i: usize| -> PyResult<Box<Expr>> {
            Ok(Box::new(Expr::parse(&parts.get_item(i)?, slots)?))
        };
        let several = || -> PyResult<Vec<Expr>> {
            parts.get_item(1)?.iter()?.map(|part| Expr::parse(&part?, slots)).collect()
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
