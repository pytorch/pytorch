//! What a region compares of a tensor it is entered with: the one
//! definition, for an argument (the guard's check), a leaf of a container
//! argument and a span's input (TreeChecks), a tensor the graph reads from
//! state (reads.rs's constants) and the Python side (`_meta` of
//! torch/_rust_inductor.py, through the class of the same name).
//!
//! Given: a Python object.
//!
//! Guarantees: `TensorKey::of(o)` is Some only for an object whose class is
//! exactly torch.Tensor or nn.Parameter and whose tensor is plain, dense and
//! strided with concrete sizes (tensor_key of csrc/tensor_ffi.h): every
//! other object has no key, so no region is traced for it and none is
//! entered with it. `key.holds(o)` is true only if `o` has a key now and it
//! is `key` in every field: the class, dtype, device, sizes, strides,
//! requires_grad, the conjugate and the negative bit, and whether it is an
//! inference tensor. No Python code runs in either.
//!
//! May not assume: anything a key does not name. Which object the tensor
//! is, where its data is and what it shares with other tensors
//! (overlap.rs), its version counter and whether it is a leaf are other
//! checks' or nobody's.

use pyo3::ffi::PyObject;
use pyo3::prelude::*;

use crate::bindings::ffi as tffi;

#[pyclass(frozen, eq, hash)]
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct TensorKey {
    // nn.Parameter, or torch.Tensor: what Python code reads of the object
    // (`x.__class__`, a module's attribute store) and no operator does.
    parameter: bool,
    facts: tffi::TensorFacts,
    sizes: Vec<i64>,
    strides: Vec<i64>,
}

impl TensorKey {
    pub unsafe fn of(obj: *mut PyObject) -> Option<TensorKey> {
        let class = pyo3::ffi::Py_TYPE(obj) as usize;
        let parameter = super::plain_tensor_types().iter().position(|t| *t == class)? == 1;
        let t = tffi::tensor_by_type(obj as usize).as_ref()?;
        let mut facts = tffi::TensorFacts::default();
        tffi::tensor_key(t, &mut facts).then(|| TensorKey { parameter, facts, sizes: tffi::tensor_sizes(t).to_vec(), strides: tffi::tensor_strides(t).to_vec() })
    }

    pub unsafe fn holds(&self, obj: *mut PyObject) -> bool {
        pyo3::ffi::Py_TYPE(obj) as usize == super::plain_tensor_types()[self.parameter as usize]
            && tffi::tensor_by_type(obj as usize).as_ref().is_some_and(|t| tffi::tensor_key_is(t, &self.facts, &self.sizes, &self.strides))
    }
}

// As a guard prints it: the fields that are not a plain tensor's follow the
// sizes and strides.
impl std::fmt::Display for TensorKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let k = &self.facts;
        write!(f, "tensor({} {} {:?} stride {:?})", tffi::dtype_name(k.dtype), tffi::device_name(k.device_type, k.device_index), self.sizes, self.strides)?;
        let flags = [(self.parameter, "Parameter"), (k.bits & 1 != 0, "requires_grad"), (k.bits & 2 != 0, "conj"), (k.bits & 4 != 0, "neg"), (k.bits & 8 != 0, "inference")];
        flags.iter().filter(|(on, _)| *on).try_for_each(|(_, name)| write!(f, " {name}"))
    }
}

#[pymethods]
impl TensorKey {
    /// The key of ``x``, or None for an object that has none (see the
    /// module).
    #[staticmethod]
    #[pyo3(name = "of")]
    fn of_object(x: &Bound<'_, PyAny>) -> Option<TensorKey> {
        unsafe { TensorKey::of(x.as_ptr()) }
    }

    /// Whether ``x`` has this key now.
    #[pyo3(name = "holds")]
    fn holds_object(&self, x: &Bound<'_, PyAny>) -> bool {
        unsafe { self.holds(x.as_ptr()) }
    }

    fn __repr__(&self) -> String {
        self.to_string()
    }
}
