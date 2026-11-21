//! The sizes a form of a graph is for.
//!
//! A block of the VM has versions, each with the guards it was specialized
//! under (`GuardCond`), and runs the first one whose guards hold. A graph
//! that was exported has forms in the same way (`graph::Exported`), and what
//! a form is for is mostly sizes: of the tensors the graph is handed. They
//! are ordinary values of the program, so one call's worth of them is a row
//! of numbers (`tensor_key`, `key_of`), and the condition under which a form
//! holds is an expression over that row (`ShapeGuard`): every number is what
//! it was when the form was made (`ShapeGuard::exact`). A guard can also
//! leave numbers open, which then only have to satisfy its conditions.

use crate::capi::{self, TcfgTensorInfo};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyTuple};

type Obj = *mut ffi::PyObject;

/// What a list of tensors is like, for the key of a graph's forms:
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
        .ok_or_else(|| PyRuntimeError::new_err("torch.vm is not initialized"))?;
    let count = tensors.len();
    let mut kinds: Vec<u8> = Vec::with_capacity(count * KIND_BYTES);
    let mut numbers: Vec<i64> = Vec::with_capacity(count * 8 + ints.len());
    let mut offsets: Vec<i64> = Vec::with_capacity(count);
    for tensor in tensors.iter() {
        match unsafe { key_of(api, tensor.as_ptr(), &mut numbers) } {
            Some((kind, offset)) => {
                kinds.extend_from_slice(&kind);
                offsets.push(offset);
            }
            None => return Err(PyTypeError::new_err("tensor_key takes plain tensors")),
        }
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

/// How many bytes say what kind a tensor is.
pub const KIND_BYTES: usize = 24;

/// What `tensor_key` says of one tensor, which may have no Python object:
/// its kind and its storage offset, with its sizes and then its strides
/// appended to `numbers`. None for what is no plain tensor.
pub unsafe fn key_of(api: &capi::TcfgApi, tensor: Obj, numbers: &mut Vec<i64>) -> Option<([u8; KIND_BYTES], i64)> {
    let mut info = TcfgTensorInfo::empty();
    if (api.tensor_info)(tensor, &mut info) == 0 {
        return None;
    }
    let rank = info.ndim as usize;
    let strided = !info.strides.is_null();
    let mut kind = [0u8; KIND_BYTES];
    kind[0..4].copy_from_slice(&info.ndim.to_ne_bytes());
    kind[4..8].copy_from_slice(&info.dtype.to_ne_bytes());
    kind[8..12].copy_from_slice(&info.device.to_ne_bytes());
    kind[12..16].copy_from_slice(&(info.requires_grad | (strided as i32) << 1).to_ne_bytes());
    kind[16..24].copy_from_slice(&info.key_set.to_ne_bytes());
    if rank > 0 {
        numbers.extend_from_slice(std::slice::from_raw_parts(info.sizes, rank));
        if strided {
            numbers.extend_from_slice(std::slice::from_raw_parts(info.strides, rank));
        }
    }
    Some((kind, info.storage_offset))
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
        self.holds_for(numbers.as_bytes(), offsets.as_bytes())
    }

    /// How many of the numbers are left open.
    #[getter]
    fn open(&self) -> usize {
        self.open.iter().filter(|open| **open).count()
    }
}

fn bytes_of(row: &[i64]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(row.as_ptr() as *const u8, std::mem::size_of_val(row)) }
}

impl ShapeGuard {
    /// The guard of a graph that leaves nothing open: it holds for the
    /// numbers it was made on.
    pub fn exact(numbers: &[i64]) -> ShapeGuard {
        let made_on = bytes_of(numbers).to_vec();
        ShapeGuard { open: vec![false; numbers.len()], made_on, offsets: 0, conditions: Vec::new() }
    }

    /// Whether the graph holds for a call with these numbers (and no
    /// storage offsets).
    pub fn holds_numbers(&self, numbers: &[i64]) -> bool {
        self.holds_for(bytes_of(numbers), &[])
    }

    /// The numbers the graph was made on.
    pub fn made_on(&self) -> Vec<i64> {
        (0..self.made_on.len() / 8).map(|index| number(&self.made_on, index)).collect()
    }

    fn holds_for(&self, numbers: &[u8], offsets: &[u8]) -> bool {
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
}
