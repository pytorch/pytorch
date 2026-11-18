//! The sizes a compiled region is called with.
//!
//! What a graph that was compiled for a region depends on in the tensors it
//! is handed is what kind they are, their sizes and how they are laid out.
//! Sizes are ordinary values of the program, so one call's worth of them is
//! a row of numbers (`tensor_key`): equal rows, for tensors of equal kinds,
//! are served by the same graph.

use crate::capi::{self, TcfgTensorInfo};
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};

/// What a list of tensors is like, for the key of a region's variants:
///
/// - their kinds: rank, element type, device, `requires_grad`, dispatch keys
///   and whether they have strides, as bytes that only compare equal for
///   tensors a graph can treat alike;
/// - their numbers: per tensor its sizes and then its strides, and after the
///   tensors the integers of `ints`, as native 64-bit integers;
/// - their storage offsets, one per tensor. Those are no part of what a
///   graph is keyed on.
#[pyfunction]
pub fn tensor_key<'py>(
    py: Python<'py>,
    tensors: &Bound<'py, PyList>,
    ints: &Bound<'py, PyList>,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    let api = capi::api()
        .ok_or_else(|| PyRuntimeError::new_err("torch.cfg_compile is not initialized"))?;
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
