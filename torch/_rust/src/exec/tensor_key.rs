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
//! requires_grad, the conjugate and the negative bit, whether it is an
//! inference tensor, and the names of the attributes the object has of
//! its own (the keys of its `__dict__`, `own_names`; an object whose
//! `__dict__` has a key that is no exact str has no key). No Python code
//! runs in either, and no `__dict__` is made for an object that has none.
//!
//! The last field is what every lookup of a name on a tensor rests on: a
//! name in the object's `__dict__` is found in front of a method and of
//! every other attribute of the class that is no data descriptor. A trace
//! that looks up a name the tensor does not have of its own reads the
//! class alone (reads.rs `attr`), and the key keeps every tensor that has
//! that name of its own out of the region.
//!
//! May not assume: anything a key does not name. Which object the tensor
//! is, where its data is and what it shares with other tensors
//! (overlap.rs), its version counter, whether it is a leaf and what its
//! own attributes hold are other checks' or nobody's.

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
    // The names of the object's own attributes, sorted.
    own: Vec<String>,
}

/// The names of the attributes `obj` has of its own: the keys of its
/// `__dict__`, sorted (none for an object without one, or with an empty
/// one). None where they cannot be told: something else sits where the
/// `__dict__` is, or a key is no exact str.
pub unsafe fn own_names(obj: *mut PyObject) -> Option<Vec<String>> {
    use pyo3::ffi;
    let d = super::ffi::tr_instance_dict(obj);
    if d.is_null() {
        return Some(vec![]);
    }
    if ffi::PyDict_Check(d) == 0 {
        return None;
    }
    let mut names = vec![];
    let (mut at, mut key, mut value) = (0, std::ptr::null_mut(), std::ptr::null_mut());
    while ffi::PyDict_Next(d, &mut at, &mut key, &mut value) != 0 {
        if ffi::Py_TYPE(key) != &raw mut ffi::PyUnicode_Type {
            return None;
        }
        let mut n: ffi::Py_ssize_t = 0;
        let text = ffi::PyUnicode_AsUTF8AndSize(key, &mut n);
        if text.is_null() {
            // (A name that is no UTF-8 text.)
            ffi::PyErr_Clear();
            return None;
        }
        names.push(String::from_utf8_lossy(std::slice::from_raw_parts(text as *const u8, n as usize)).into_owned());
    }
    names.sort();
    Some(names)
}

impl TensorKey {
    pub unsafe fn of(obj: *mut PyObject) -> Option<TensorKey> {
        let class = pyo3::ffi::Py_TYPE(obj) as usize;
        let parameter = super::plain_tensor_types().iter().position(|t| *t == class)? == 1;
        let t = tffi::tensor_by_type(obj as usize).as_ref()?;
        let mut facts = tffi::TensorFacts::default();
        if !tffi::tensor_key(t, &mut facts) {
            return None;
        }
        // (A tensor that has a key has sizes and strides to read.)
        Some(TensorKey { parameter, facts, sizes: tffi::tensor_sizes(t).ok()?.to_vec(), strides: tffi::tensor_strides(t).ok()?.to_vec(), own: own_names(obj)? })
    }

    pub unsafe fn holds(&self, obj: *mut PyObject) -> bool {
        pyo3::ffi::Py_TYPE(obj) as usize == super::plain_tensor_types()[self.parameter as usize]
            && own_names(obj).as_ref() == Some(&self.own)
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
        flags.iter().filter(|(on, _)| *on).try_for_each(|(_, name)| write!(f, " {name}"))?;
        if self.own.is_empty() { Ok(()) } else { write!(f, " attributes {}", self.own.join(",")) }
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

    /// Whether the tensor has the conjugate or the negative bit: its memory
    /// is not its values, and code made to read memory takes no such tensor.
    #[getter]
    fn marked(&self) -> bool {
        self.facts.bits & 6 != 0
    }

    fn __repr__(&self) -> String {
        self.to_string()
    }
}
