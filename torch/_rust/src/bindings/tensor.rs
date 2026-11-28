use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::ffi;

pub type Tensor = ffi::Tensor;

impl Tensor {
    // Err for a tensor whose sizes cannot be read as a list of ints (a
    // nested tensor, symbolic sizes, a Python-dispatch tensor that raises).
    pub fn sizes(&self) -> Result<&[i64], cxx::Exception> {
        ffi::tensor_sizes(self)
    }

    pub fn strides(&self) -> Result<&[i64], cxx::Exception> {
        ffi::tensor_strides(self)
    }

    pub fn is_contiguous(&self) -> bool {
        ffi::tensor_is_contiguous(self)
    }

    pub fn is_cpu(&self) -> bool {
        ffi::tensor_is_cpu(self)
    }

    pub fn requires_grad(&self) -> bool {
        ffi::tensor_requires_grad(self)
    }

    pub fn is_leaf(&self) -> bool {
        ffi::tensor_is_leaf(self)
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for &'a Tensor {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let ptr = unsafe { ffi::tensor_from_pyobject(ob.as_ptr() as usize) };
        if ptr.is_null() {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "expected torch.Tensor",
            ));
        }
        // The at::Tensor lives in the THPVariable C struct owned by `ob`'s
        // Python object. That object is kept alive by reference counting via
        // `ob`'s strong reference, not by the GIL (this module declares
        // gil_used = false).
        Ok(unsafe { &*ptr })
    }
}

// SAFETY: type_check accepts exactly the objects THPVariable_Check accepts,
// i.e. instances of torch.Tensor and its subclasses.
unsafe impl pyo3::PyTypeCheck for Tensor {
    const NAME: &'static str = "torch.Tensor";

    fn type_check(object: &Bound<'_, PyAny>) -> bool {
        let ptr = unsafe { ffi::tensor_from_pyobject(object.as_ptr() as usize) };
        !ptr.is_null()
    }

    fn classinfo_object(py: Python<'_>) -> Bound<'_, PyAny> {
        py.import("torch")
            .and_then(|m| m.getattr("Tensor"))
            .expect("torch.Tensor")
    }
}

/// What each function of the bridge that takes a tensor answers of `obj`,
/// by the function's name: for the tests of the bridge, which hand it
/// tensors whose questions raise in C++. `a_pending_error_is_kept`: whether
/// a Python error that is pending when the functions that may run Python
/// code are called is still the pending error after them.
#[pyfunction]
pub fn _bridge_answers<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    let p = obj.as_ptr() as usize;
    let (by_type, from) = unsafe { (ffi::tensor_by_type(p), ffi::tensor_from_pyobject(p)) };
    out.set_item("tensor_by_type", !by_type.is_null())?;
    out.set_item("tensor_from_pyobject", !from.is_null())?;
    let Some(t) = (unsafe { from.as_ref() }) else { return Ok(out) };
    let mut facts = ffi::TensorFacts::default();
    out.set_item("tensor_key", ffi::tensor_key(t, &mut facts))?;
    let (sizes, strides) = (ffi::tensor_sizes(t).ok().map(<[i64]>::to_vec), ffi::tensor_strides(t).ok().map(<[i64]>::to_vec));
    out.set_item("tensor_key_is", ffi::tensor_key_is(t, &facts, sizes.as_deref().unwrap_or(&[]), strides.as_deref().unwrap_or(&[])))?;
    out.set_item("tensor_sizes", sizes)?;
    out.set_item("tensor_strides", strides)?;
    out.set_item("tensor_span", ffi::tensor_span(t, &mut ffi::TensorSpan::default()))?;
    out.set_item("tensor_home", ffi::tensor_home(t, &mut 0) != 0)?;
    out.set_item("tensor_version", ffi::tensor_version(t))?;
    out.set_item("tensor_is_cow", ffi::tensor_is_cow(t))?;
    out.set_item("tensor_is_contiguous", ffi::tensor_is_contiguous(t))?;
    out.set_item("tensor_is_cpu", ffi::tensor_is_cpu(t))?;
    out.set_item("tensor_is_leaf", ffi::tensor_is_leaf(t))?;
    out.set_item("tensor_requires_grad", ffi::tensor_requires_grad(t))?;
    let mut meta = ffi::TensorMeta::default();
    ffi::tensor_meta(t, &mut meta);
    out.set_item("tensor_meta", (meta.defined, meta.has_device, meta.has_shape))?;
    let kept = unsafe {
        pyo3::ffi::PyErr_SetString(pyo3::ffi::PyExc_KeyError, c"pending".as_ptr());
        let _ = (ffi::tensor_from_pyobject(p), ffi::tensor_is_contiguous(t), ffi::tensor_is_cpu(t), ffi::tensor_sizes(t).is_ok(), ffi::region_not_neutral());
        let kept = pyo3::ffi::PyErr_ExceptionMatches(pyo3::ffi::PyExc_KeyError) != 0;
        pyo3::ffi::PyErr_Clear();
        kept
    };
    out.set_item("a_pending_error_is_kept", kept)?;
    Ok(out)
}
