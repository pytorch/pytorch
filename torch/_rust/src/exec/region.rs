// Regions: a pure block's version compiled as one graph.
//
// Once a pure block's (unguarded beyond the optimizer's own checks) version
// has run often enough, its next run is traced: torch._rust_inductor's
// trace_region runs the body (through _region_body) under make_fx on the
// real inputs, and hands the aten graph to a backend (Inductor). The block
// then gets a version in front of the others whose guard adds what the trace
// depended on (each tensor's dtype, device, sizes, strides and requires_grad,
// each scalar's value, each object's identity, grad mode) and whose body is
// one Op::Region: the compiled graph, called on the tensors among the block's
// params and inputs. A run whose guard fails falls through to the version
// the trace came from, which traces again (another variant) when it has run
// often enough again, up to MAX_REGIONS variants per block.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use pyo3::ffi;
use pyo3::ffi::PyObject;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::cfg::ir::{RegionOut, RegionSpec};

static ENABLED: AtomicBool = AtomicBool::new(false);
static AFTER: AtomicU32 = AtomicU32::new(2);
pub const MAX_REGIONS: usize = 8;

/// Compile pure blocks into regions (see exec/region.rs) once a version has
/// run `after` times.
#[pyfunction]
#[pyo3(signature = (on, after=2))]
pub fn _set_cfg_regions(on: bool, after: u32) {
    ENABLED.store(on, Ordering::Relaxed);
    AFTER.store(after.max(1), Ordering::Relaxed);
}

#[inline]
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

// Whether block b's version (the last, the one traces are made from) should
// be traced on this run.
pub fn due(runs: &AtomicU32, off: &AtomicBool) -> bool {
    !off.load(Ordering::Relaxed) && runs.fetch_add(1, Ordering::Relaxed) + 1 == AFTER.load(Ordering::Relaxed)
}

// Runs the body being traced on the given args (borrowed); Ok(its outputs,
// owned) or Err with the body's exception set.
pub type Body<'a> = dyn FnMut(&[*mut PyObject]) -> Result<Vec<*mut PyObject>, ()> + 'a;

thread_local! {
    static BODY: Cell<Option<*mut Body<'static>>> = const { Cell::new(None) };
}

fn null_sentinel(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    py.import("torch._rust_inductor")?.getattr("NULL")
}

/// Runs the body of the block being traced (see torch._rust_inductor), once.
#[pyfunction]
#[pyo3(signature = (*args))]
pub fn _region_body<'py>(py: Python<'py>, args: &Bound<'py, PyTuple>) -> PyResult<Bound<'py, PyTuple>> {
    let Some(body) = BODY.take() else {
        return Err(pyo3::exceptions::PyRuntimeError::new_err("no region body is being traced"));
    };
    let null = null_sentinel(py)?;
    let objs: Vec<*mut PyObject> = args.iter().map(|a| if a.is(&null) { std::ptr::null_mut() } else { a.as_ptr() }).collect();
    let outs = unsafe { (*body)(&objs) };
    match outs {
        Ok(outs) => {
            let items: Vec<Bound<'py, PyAny>> = outs
                .into_iter()
                .map(|o| unsafe { Bound::from_owned_ptr_or_opt(py, o) }.unwrap_or_else(|| null.clone()))
                .collect();
            PyTuple::new(py, items)
        }
        Err(()) => Err(PyErr::fetch(py)),
    }
}

// What a trace produced: the spec, and the checks to add to the guard, as
// (position among the args or None, check tuple) for mod.rs's parse_check.
pub struct Traced {
    pub spec: RegionSpec,
    pub checks: Vec<Py<PyTuple>>,
}

// Traces a run of a block's version. `args`: its params then inputs
// (borrowed, NULL allowed), of which those at `unused` aren't read by the
// version (NULL here); `body` runs it. Ok(None): not worth a region.
pub unsafe fn trace(name: &str, args: &[*mut PyObject], unused: &[u32], body: &mut Body<'_>) -> PyResult<Option<Traced>> {
    let py = Python::assume_attached();
    let null = null_sentinel(py)?;
    let items: Vec<Bound<'_, PyAny>> =
        args.iter().map(|a| Bound::from_borrowed_ptr_or_opt(py, *a).unwrap_or_else(|| null.clone())).collect();
    let args = PyTuple::new(py, items)?;
    let body: *mut Body<'_> = body;
    BODY.set(Some(std::mem::transmute::<*mut Body<'_>, *mut Body<'static>>(body)));
    let r = py.import("torch._rust_inductor").and_then(|m| m.call_method1("trace_region", (name, args, unused.to_vec())));
    BODY.set(None);
    let r = r?;
    if r.is_none() {
        return Ok(None);
    }
    let (func, tensor_args, outputs, objects, checks): (Bound<'_, PyAny>, Vec<u32>, Vec<Bound<'_, PyAny>>, Vec<Bound<'_, PyAny>>, Vec<Bound<'_, PyTuple>>) =
        r.extract()?;
    let outputs = outputs.iter().map(parse_out).collect::<PyResult<Vec<_>>>()?;
    Ok(Some(Traced {
        spec: RegionSpec {
            func: func.unbind(),
            tensor_args,
            outputs,
            objects: objects.into_iter().map(Bound::unbind).collect(),
            name: name.to_string(),
        },
        checks: checks.into_iter().map(Bound::unbind).collect(),
    }))
}

fn parse_out(o: &Bound<'_, PyAny>) -> PyResult<RegionOut> {
    let t = o.cast::<PyTuple>()?;
    let kind: String = t.get_item(0)?.extract()?;
    Ok(match kind.as_str() {
        "arg" => RegionOut::Arg(t.get_item(1)?.extract()?),
        "out" => RegionOut::Out(t.get_item(1)?.extract()?),
        "obj" => RegionOut::Obj(t.get_item(1)?.extract()?),
        "null" => RegionOut::Null,
        "tuple" | "list" => {
            let items = t.get_item(1)?.try_iter()?.map(|x| parse_out(&x?)).collect::<PyResult<Vec<_>>>()?;
            if kind == "tuple" { RegionOut::Tuple(items) } else { RegionOut::List(items) }
        }
        _ => return Err(pyo3::exceptions::PyValueError::new_err(format!("bad region output {o}"))),
    })
}

// Runs an Op::Region on `args` (borrowed): the block's outputs (owned), or
// Err with an exception set.
pub unsafe fn call(spec: &RegionSpec, args: &[*mut PyObject]) -> Result<Vec<*mut PyObject>, ()> {
    let list = ffi::PyList_New(spec.tensor_args.len() as ffi::Py_ssize_t);
    if list.is_null() {
        return Err(());
    }
    for (i, k) in spec.tensor_args.iter().enumerate() {
        let a = args[*k as usize];
        ffi::Py_INCREF(a);
        ffi::PyList_SET_ITEM(list, i as ffi::Py_ssize_t, a);
    }
    // The compiled graph's Python wrappers run on CPython.
    let r = super::outside_executor(|| ffi::PyObject_CallOneArg(spec.func.as_ptr(), list));
    ffi::Py_DECREF(list);
    if r.is_null() {
        return Err(());
    }
    let outs = if ffi::PyList_CheckExact(r) != 0 || ffi::PyTuple_CheckExact(r) != 0 { r } else { ffi::PySequence_Fast(r, c"region outputs".as_ptr()) };
    if outs.is_null() {
        ffi::Py_DECREF(r);
        return Err(());
    }
    let item = |k: u32| -> *mut PyObject {
        if ffi::PyList_CheckExact(outs) != 0 { ffi::PyList_GET_ITEM(outs, k as ffi::Py_ssize_t) } else { ffi::PyTuple_GET_ITEM(outs, k as ffi::Py_ssize_t) }
    };
    let n = ffi::PySequence_Size(outs);
    unsafe fn make(spec: &RegionSpec, o: &RegionOut, args: &[*mut PyObject], item: &dyn Fn(u32) -> *mut PyObject, n: isize) -> *mut PyObject {
        match o {
            RegionOut::Arg(i) => {
                let a = args[*i as usize];
                if !a.is_null() {
                    ffi::Py_INCREF(a);
                }
                a
            }
            RegionOut::Out(k) => {
                if *k as isize >= n {
                    ffi::PyErr_SetString(ffi::PyExc_RuntimeError, c"CFG executor: a region returned too few outputs".as_ptr());
                    return std::ptr::null_mut();
                }
                let x = item(*k);
                ffi::Py_INCREF(x);
                x
            }
            RegionOut::Obj(i) => {
                let x = spec.objects[*i as usize].as_ptr();
                ffi::Py_INCREF(x);
                x
            }
            RegionOut::Null => std::ptr::null_mut(),
            RegionOut::Tuple(items) | RegionOut::List(items) => {
                let tuple = matches!(o, RegionOut::Tuple(_));
                let c = if tuple { ffi::PyTuple_New(items.len() as isize) } else { ffi::PyList_New(items.len() as isize) };
                for (i, it) in items.iter().enumerate() {
                    let x = make(spec, it, args, item, n);
                    if x.is_null() {
                        ffi::Py_DECREF(c);
                        return std::ptr::null_mut();
                    }
                    if tuple {
                        ffi::PyTuple_SET_ITEM(c, i as isize, x);
                    } else {
                        ffi::PyList_SET_ITEM(c, i as isize, x);
                    }
                }
                c
            }
        }
    }
    let mut result = Vec::with_capacity(spec.outputs.len());
    let mut ok = true;
    for o in &spec.outputs {
        let x = make(spec, o, args, &item, n);
        if x.is_null() && !matches!(o, RegionOut::Null | RegionOut::Arg(_)) {
            ok = false;
            break;
        }
        result.push(x);
    }
    if outs != r {
        ffi::Py_DECREF(outs);
    }
    ffi::Py_DECREF(r);
    if !ok {
        result.into_iter().for_each(|x| ffi::Py_XDECREF(x));
        return Err(());
    }
    Ok(result)
}
