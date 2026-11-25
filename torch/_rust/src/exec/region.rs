// Regions: a pure block's version compiled as one graph.
//
// Once a pure block's (unguarded beyond the optimizer's own checks) version
// has run often enough, its next run is traced: torch._rust_inductor's
// trace_region runs the body (through _region_body) under make_fx on the
// real inputs, and hands the aten graph to a backend (Inductor). The block
// then gets a version in front of the others whose guard adds what the trace
// depended on (each tensor's dtype, device, sizes, strides and requires_grad,
// each scalar's value, each object's identity, grad mode) and whose body is
// one Op::Region: the compiled graph, called (from Python, which checks the
// containers among them) on the block's params and inputs, once what the
// trace read from elsewhere (see reads.rs) is seen to be as it was. A run whose guard fails falls through to the version
// the trace came from, which traces again (another variant) when it has run
// often enough again, up to max_regions() variants per block.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};

use pyo3::ffi;
use pyo3::ffi::PyObject;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::cfg::ir::{RegionOut, RegionSpec};
use crate::exec::types::Slot;

static ENABLED: AtomicBool = AtomicBool::new(false);
static OBSERVE: AtomicBool = AtomicBool::new(false);
static AFTER: AtomicU32 = AtomicU32::new(2);
static MAX_REGIONS: AtomicUsize = AtomicUsize::new(4);
static CHECK_READS: AtomicBool = AtomicBool::new(true);

/// Whether a region is entered only while what its trace read is as it was
/// (see exec/reads.rs). On, except to measure what the check costs: a region
/// that isn't checked computes with what its trace read.
#[pyfunction]
pub fn _set_cfg_region_reads(on: bool) {
    CHECK_READS.store(on, Ordering::Relaxed);
}

/// Compile pure blocks into regions (see exec/region.rs) once a version has
/// run `after` times, up to `variants` per block (a loop body over different
/// modules needs one per module).
#[pyfunction]
#[pyo3(signature = (on, after=2, variants=4))]
pub fn _set_cfg_regions(on: bool, after: u32, variants: usize) {
    ENABLED.store(on, Ordering::Relaxed);
    AFTER.store(after.max(1), Ordering::Relaxed);
    MAX_REGIONS.store(variants.max(1), Ordering::Relaxed);
}

/// Record the types of the args of the blocks and spans that would be
/// traced, for their traces (torch.vm's input_types), whether or not regions
/// are on.
#[pyfunction]
pub fn _set_cfg_region_observe(on: bool) {
    OBSERVE.store(on, Ordering::Relaxed);
}

#[inline]
pub fn observing() -> bool {
    OBSERVE.load(Ordering::Relaxed)
}

#[inline]
pub fn max_regions() -> usize {
    MAX_REGIONS.load(Ordering::Relaxed)
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

// Joins a run's args (borrowed, NULL allowed) into their types.
pub unsafe fn observe(types: &mut Vec<Slot>, args: &[*mut PyObject]) {
    types.resize_with(args.len(), Slot::default);
    for (slot, a) in types.iter_mut().zip(args) {
        if !slot.skips(*a) {
            slot.observe(*a, false);
        }
    }
}

// Runs the body being traced on the given args (borrowed); Ok(its outputs,
// owned) or Err with the body's exception set.
pub type Body<'a> = dyn FnMut(&[*mut PyObject]) -> Result<Vec<*mut PyObject>, ()> + 'a;

thread_local! {
    static BODY: Cell<Option<*mut Body<'static>>> = const { Cell::new(None) };
    static TRACING: Cell<bool> = const { Cell::new(false) };
}

// Whether a trace is running on this thread. Regions don't run then (the
// trace wouldn't see their ops), nor are others traced.
#[inline]
pub fn tracing() -> bool {
    TRACING.get()
}

// While a region is traced, the floats it computes with are instances of a
// float subclass (torch._rust_inductor._Scalar), so that the arithmetic on
// them is recorded. The traced run is also the call's own run, so what it
// leaves behind (a block's outputs; a span's locals, stack values and return
// value) is given to the program as floats: `o` (borrowed, NULL allowed) with
// each such instance replaced by the float it is, in `o` itself and in the
// exact tuples, lists and dicts it holds. A new reference, to `o` where there
// was nothing to replace.
pub unsafe fn plain(o: *mut PyObject) -> *mut PyObject {
    if o.is_null() {
        return o;
    }
    let py = Python::assume_attached();
    let made = py
        .import("torch._rust_inductor")
        .and_then(|m| m.getattr("_plain_result"))
        .and_then(|f| f.call1((Bound::from_borrowed_ptr(py, o),)));
    match made {
        Ok(x) => x.into_ptr(),
        Err(_) => {
            ffi::PyErr_Clear();
            ffi::Py_INCREF(o);
            o
        }
    }
}

fn null_sentinel(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    static NULL: pyo3::sync::PyOnceLock<Py<PyAny>> = pyo3::sync::PyOnceLock::new();
    NULL.get_or_try_init(py, || Ok::<_, PyErr>(py.import("torch._rust_inductor")?.getattr("NULL")?.unbind())).map(|n| n.bind(py).clone())
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
    // The trace runs outside the executor; the body runs in it again, so
    // that the frames it enters are the executor's and what they read is
    // recorded (see reads.rs).
    let outs = {
        let _exec = super::ExecGuard::new();
        let _recording = super::reads::Recording::start();
        unsafe { (*body)(&objs) }
    };
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
// (borrowed, NULL allowed), of which those at `unread` aren't read by the
// version's instrs (NULL if not used at all), and `types` their types over
// the runs observed (see observing); `body` runs it. Ok(None): not worth a
// region. With `span` (a span's run, which no guard checks first), the region
// makes all its checks itself.
pub unsafe fn trace(name: &str, args: &[*mut PyObject], types: &[Slot], unread: &[u32], span: bool, body: &mut Body<'_>) -> PyResult<Option<Traced>> {
    let py = Python::assume_attached();
    let null = null_sentinel(py)?;
    let items: Vec<Bound<'_, PyAny>> =
        args.iter().map(|a| Bound::from_borrowed_ptr_or_opt(py, *a).unwrap_or_else(|| null.clone())).collect();
    let args = PyTuple::new(py, items)?;
    let types = (0..args.len())
        .map(|i| match types.get(i).and_then(|s| s.ty.as_ref()) {
            Some(t) => t.to_py(py),
            None => Ok(py.None().into_bound(py)),
        })
        .collect::<PyResult<Vec<_>>>()?;
    let body: *mut Body<'_> = body;
    BODY.set(Some(std::mem::transmute::<*mut Body<'_>, *mut Body<'static>>(body)));
    TRACING.set(true);
    super::reads::begin();
    let r = py.import("torch._rust_inductor").and_then(|m| m.call_method1("trace_region", (name, args, unread.to_vec(), span, types)));
    super::reads::end();
    TRACING.set(false);
    BODY.set(None);
    let r = r?;
    if r.is_none() {
        return Ok(None);
    }
    let (func, outputs, objects, checks, reads): (Bound<'_, PyAny>, Vec<Bound<'_, PyAny>>, Vec<Bound<'_, PyAny>>, Vec<Bound<'_, PyTuple>>, Option<Py<super::reads::Reads>>) =
        r.extract()?;
    let outputs = outputs.iter().map(parse_out).collect::<PyResult<Vec<_>>>()?;
    Ok(Some(Traced {
        spec: RegionSpec {
            func: func.unbind(),
            outputs,
            objects: objects.into_iter().map(Bound::unbind).collect(),
            name: name.to_string(),
            reads,
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
        "dict" => {
            let keys: Vec<u32> = t.get_item(1)?.extract()?;
            let items = t.get_item(2)?.try_iter()?.map(|x| parse_out(&x?)).collect::<PyResult<Vec<_>>>()?;
            RegionOut::Dict(keys, items)
        }
        "iter" => RegionOut::Iter(Box::new(parse_out(&t.get_item(1)?)?)),
        "call" => {
            let items = t.get_item(2)?.try_iter()?.map(|x| parse_out(&x?)).collect::<PyResult<Vec<_>>>()?;
            RegionOut::Call(t.get_item(1)?.extract()?, items)
        }
        _ => return Err(pyo3::exceptions::PyValueError::new_err(format!("bad region output {o}"))),
    })
}

// Runs an Op::Region on `args` (borrowed: the block's params, then the
// version's inputs): Some(the block's outputs, owned), None if the region's
// own checks failed (run the version it was traced from), or Err with an
// exception set.
pub unsafe fn call(spec: &RegionSpec, args: &[*mut PyObject]) -> Result<Option<Vec<*mut PyObject>>, ()> {
    let py = Python::assume_attached();
    // Before anything of the region runs.
    if let Some(reads) = &spec.reads
        && CHECK_READS.load(Ordering::Relaxed)
        && !reads.get().check()
    {
        if std::env::var_os("TORCH_RUST_REGION_DEBUG").is_some() {
            eprintln!("region {}: miss on {}", spec.name, reads.get().last());
        }
        return Ok(None);
    }
    let Ok(null) = null_sentinel(py) else { return Err(()) };
    let list = ffi::PyList_New(args.len() as ffi::Py_ssize_t);
    if list.is_null() {
        return Err(());
    }
    for (i, a) in args.iter().enumerate() {
        let a = if a.is_null() { null.as_ptr() } else { *a };
        ffi::Py_INCREF(a);
        ffi::PyList_SET_ITEM(list, i as ffi::Py_ssize_t, a);
    }
    // The compiled graph's Python wrappers run on CPython.
    let r = super::outside_executor(|| ffi::PyObject_CallOneArg(spec.func.as_ptr(), list));
    ffi::Py_DECREF(list);
    if r.is_null() {
        return Err(());
    }
    if r == ffi::Py_None() {
        ffi::Py_DECREF(r);
        return Ok(None);
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
            RegionOut::Call(f, items) => {
                let list = ffi::PyList_New(items.len() as isize);
                for (i, it) in items.iter().enumerate() {
                    let x = make(spec, it, args, item, n);
                    if x.is_null() {
                        ffi::Py_DECREF(list);
                        return std::ptr::null_mut();
                    }
                    ffi::PyList_SET_ITEM(list, i as isize, x);
                }
                let r = ffi::PyObject_CallOneArg(spec.objects[*f as usize].as_ptr(), list);
                ffi::Py_DECREF(list);
                r
            }
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
            RegionOut::Dict(keys, items) => {
                let d = ffi::PyDict_New();
                for (k, it) in keys.iter().zip(items) {
                    let x = make(spec, it, args, item, n);
                    if x.is_null() || ffi::PyDict_SetItem(d, spec.objects[*k as usize].as_ptr(), x) < 0 {
                        ffi::Py_XDECREF(x);
                        ffi::Py_DECREF(d);
                        return std::ptr::null_mut();
                    }
                    ffi::Py_DECREF(x);
                }
                d
            }
            RegionOut::Iter(items) => {
                let c = make(spec, items, args, item, n);
                if c.is_null() {
                    return c;
                }
                let it = ffi::PyObject_GetIter(c);
                ffi::Py_DECREF(c);
                it
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
    Ok(Some(result))
}
