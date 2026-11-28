//! What a builtin builds at a traced call that no torch op stands for.
//!
//! A region is the torch ops of one traced run, made again. A call of C
//! code can hand out a new tensor with more to it than the ops that ran:
//!   - a leaf that requires grad (`torch.zeros(2, requires_grad=True)`, any
//!     factory with the keyword, `torch.tensor(data, requires_grad=True)`):
//!     the binding sets the flag on its result after the operator. The
//!     region's graph makes the tensor without it;
//!   - a tensor filled from Python data that holds tensors
//!     (`torch.tensor([a, b])`, torch.as_tensor, torch.asarray,
//!     Tensor.new_tensor, Tensor.new and the tensor classes called with a
//!     list): C++ reads the value of each tensor in the data with every
//!     mode off, so the trace sees no read, and the result is a constant
//!     of the graph.
//!
//! Interface. `refusal(callee, handed, result)` is asked by reads.rs of
//! each call the traced run makes that the executor does not run itself (a
//! builtin, a class, a functools.partial), once it returned (`handed`: the
//! positional arguments, then the keyword values; `spread` gives them for
//! a starred call): Some(why) refuses the trace. `init()` is called where
//! a trace starts.
//! What it may not assume: that the calls observed at the instruction were
//! handed the same kinds of data or made the same kind of tensor (an
//! observed read of a tensor's value ends a region where it is, see
//! value_reads.rs: this is for the call that is the first to make one);
//! that a tensor with one reference was made by the call (one taken out of
//! a container has one too: that errs towards a refusal, and such a call
//! is a write).
use std::sync::OnceLock;

use pyo3::ffi;
use pyo3::ffi::PyObject;
use pyo3::prelude::*;

use super::ffi::tr_bound_function;
use crate::bindings::ffi as tffi;

// torch.Tensor's C base class, then the callables that fill a new tensor
// from the numbers of Python data: by address (a method by its descriptor).
static FILLERS: OnceLock<Vec<usize>> = OnceLock::new();

// Looks the callables up: called where a trace starts, outside the executor
// (not a read of the body's).
pub(super) fn init() {
    FILLERS.get_or_init(|| {
        let py = unsafe { Python::assume_attached() };
        let found = || -> PyResult<Vec<usize>> {
            let torch = py.import("torch")?;
            let base = torch.getattr("_C")?.getattr("TensorBase")?;
            let methods = base.getattr("__dict__")?;
            let mut out = vec![base.as_ptr() as usize];
            for name in ["tensor", "as_tensor", "asarray"] {
                out.push(torch.getattr(name)?.as_ptr() as usize);
            }
            for name in ["new_tensor", "new"] {
                out.push(methods.get_item(name)?.as_ptr() as usize);
            }
            for class in torch.getattr("_tensor_classes")?.try_iter()? {
                out.push(class?.as_ptr() as usize);
            }
            Ok(out)
        };
        found().unwrap_or_default()
    });
}

unsafe fn fills(callee: *mut PyObject) -> bool {
    let Some(known) = FILLERS.get() else { return false };
    let Some(base) = known.first() else { return false };
    known[1..].contains(&(callee as usize))
        || known[1..].contains(&(tr_bound_function(callee) as usize))
        || (ffi::PyType_Check(callee) != 0 && ffi::PyType_IsSubtype(callee as *mut ffi::PyTypeObject, *base as *mut ffi::PyTypeObject) != 0)
}

// Whether `o`, or what a list or a tuple in it holds, is a tensor.
unsafe fn holds_tensor(o: *mut PyObject, depth: u8) -> bool {
    if o.is_null() || depth > 16 {
        return false;
    }
    let items = |n: isize, at: unsafe fn(*mut PyObject, isize) -> *mut PyObject| (0..n).any(|i| holds_tensor(at(o, i), depth + 1));
    if ffi::PyList_Check(o) != 0 {
        items(ffi::PyList_GET_SIZE(o), ffi::PyList_GET_ITEM)
    } else if ffi::PyTuple_Check(o) != 0 {
        items(ffi::PyTuple_GET_SIZE(o), ffi::PyTuple_GET_ITEM)
    } else {
        !tffi::tensor_by_type(o as usize).is_null()
    }
}

// The arguments of a starred call (borrowed): the items of `args` (a tuple
// or a list), then the values of `kwargs` (a dict or NULL).
pub(super) unsafe fn spread(args: *mut PyObject, kwargs: *mut PyObject) -> Vec<*mut PyObject> {
    let mut out = vec![];
    if ffi::PyTuple_Check(args) != 0 {
        out.extend((0..ffi::PyTuple_GET_SIZE(args)).map(|i| ffi::PyTuple_GET_ITEM(args, i)));
    } else if ffi::PyList_Check(args) != 0 {
        out.extend((0..ffi::PyList_GET_SIZE(args)).map(|i| ffi::PyList_GET_ITEM(args, i)));
    }
    if !kwargs.is_null() && ffi::PyDict_Check(kwargs) != 0 {
        let (mut pos, mut key, mut value) = (0, std::ptr::null_mut(), std::ptr::null_mut());
        while ffi::PyDict_Next(kwargs, &mut pos, &mut key, &mut value) != 0 {
            out.push(value);
        }
    }
    out
}

// `callee` was handed `handed` and just returned `result` (not NULL) to the
// traced run.
pub(super) unsafe fn refusal(callee: *mut PyObject, handed: &[*mut PyObject], result: *mut PyObject) -> Option<String> {
    let t = tffi::tensor_by_type(result as usize).as_ref()?;
    // The executor's reference is the only one: the call made the tensor.
    if ffi::Py_REFCNT(result) == 1 && t.requires_grad() && t.is_leaf() {
        return Some("it makes a tensor that requires grad, which a region makes without".into());
    }
    // A tensor that is the data itself is copied by operators.
    let inside = |o: &*mut PyObject| (ffi::PyList_Check(*o) != 0 || ffi::PyTuple_Check(*o) != 0) && holds_tensor(*o, 0);
    (fills(callee) && handed.iter().any(inside)).then(|| "it makes a tensor from the values of the tensors in a list, which no operator reads".into())
}
