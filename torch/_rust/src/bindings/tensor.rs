use pyo3::prelude::*;

use super::ffi;

pub type Tensor = ffi::Tensor;

impl Tensor {
    pub fn dim(&self) -> i64 {
        ffi::tensor_dim(self)
    }

    pub fn numel(&self) -> i64 {
        ffi::tensor_numel(self)
    }

    pub fn size(&self, dim: i64) -> i64 {
        ffi::tensor_size_at(self, dim)
    }

    pub fn stride(&self, dim: i64) -> i64 {
        ffi::tensor_stride_at(self, dim)
    }

    pub fn sizes(&self) -> &[i64] {
        ffi::tensor_sizes(self)
    }

    pub fn strides(&self) -> &[i64] {
        ffi::tensor_strides(self)
    }

    pub fn is_contiguous(&self) -> bool {
        ffi::tensor_is_contiguous(self)
    }

    pub fn is_cpu(&self) -> bool {
        ffi::tensor_is_cpu(self)
    }

    pub fn is_cuda(&self) -> bool {
        ffi::tensor_is_cuda(self)
    }

    pub fn defined(&self) -> bool {
        ffi::tensor_defined(self)
    }

    pub fn requires_grad(&self) -> bool {
        ffi::tensor_requires_grad(self)
    }
}

impl<'py> FromPyObject<'py> for &'py Tensor {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
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

impl pyo3::PyTypeCheck for Tensor {
    const NAME: &'static str = "torch.Tensor";

    fn type_check(object: &Bound<'_, PyAny>) -> bool {
        let ptr = unsafe { ffi::tensor_from_pyobject(object.as_ptr() as usize) };
        !ptr.is_null()
    }
}
