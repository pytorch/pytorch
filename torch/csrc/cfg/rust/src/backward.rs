//! The backward of a graph, derived in the engine.
//!
//! A form whose inputs need a gradient is a `Staged`: what the exporter hands
//! the VM in place of a callable that a pass made. For its first calls the
//! form runs on eager's tape, one operation of the exported graph after the
//! other, each under the tape of the C API, which hands over the kernels the
//! operation ran with their operands and results: the forward of the joint
//! graph (`joint.rs`). Behind the forward eager's own autograd nodes for
//! those kernels are called one after the other, latest first, as the
//! autograd engine calls them, again under the tape: what they run is the
//! backward. What the nodes gave is compared, bit for bit, with what the
//! autograd engine gives for the same tape. After two calls that came out
//! the same graph, the form runs its forward as a straight-line program
//! below the autograd keys and gives its results one node (`GraphBackward`,
//! in cfg.cpp), which runs the backward program for the results that come
//! with a gradient. A form that cannot be taken down keeps eager's tape.
//!
//! What Python still does for this is the export boundary: where the list
//! of passes has a pass over graphs, the two programs are made
//! `torch.fx` graphs for the passes (`torch/vm/_export.py`).

use crate::capi::{self, TcfgBackwardApi, TcfgNodeInfo, TcfgTensorInfo, TCFG_NODE_LEAF, TCFG_NODE_PYTHON, TCFG_RUN_BELOW_AUTOGRAD, TCFG_RUN_ON_TAPE};
use crate::joint::{Joint, Value};
use crate::pyffi::{self, Obj, Once};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict, PyList, PyTuple, PyWeakrefMethods, PyWeakrefReference};
use std::collections::{HashMap, HashSet};
use std::ffi::{c_void, CStr};
use std::os::raw::c_char;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

/// Why a form, or a call of one, does not get the engine's backward.
enum Stop {
    /// The form keeps eager's tape, for this reason.
    Refused(String),
    /// A view that a result is made of is read in the graph too, first by
    /// this operation of the forward: the form is cut in front of it.
    Cut(usize),
    /// Something raised: the call raises it.
    Raised(PyErr),
}

impl From<PyErr> for Stop {
    fn from(error: PyErr) -> Stop {
        Stop::Raised(error)
    }
}

/// What the joint graph refuses is a reason; what else it raises is raised.
fn joint_says<T>(py: Python<'_>, result: PyResult<T>) -> Result<T, Stop> {
    result.map_err(|error| {
        if error.is_instance_of::<PyValueError>(py) {
            Stop::Refused(unsafe { crate::graph::message(error.value(py).as_ptr()) })
        } else {
            Stop::Raised(error)
        }
    })
}

fn refused<T>(why: impl Into<String>) -> Result<T, Stop> {
    Err(Stop::Refused(why.into()))
}

// ---------------------------------------------------------------------------
// torch, as far as the engine asks it something.

static TORCH: Once<Py<PyModule>> = Once::new();
static TENSOR: Once<Py<PyAny>> = Once::new();

/// The module as the interpreter keeps it (`pyffi::module`): it is not
/// imported here.
pub(crate) fn torch(py: Python<'_>) -> PyResult<&Bound<'_, PyModule>> {
    if let Some(found) = TORCH.get() {
        return Ok(found.bind(py));
    }
    let found = unsafe { pyffi::module(c"torch") };
    if found.is_null() {
        return Err(PyRuntimeError::new_err("torch.vm: the module torch is not there"));
    }
    TORCH.set(unsafe { Bound::from_borrowed_ptr(py, found).cast_into_unchecked::<PyModule>().unbind() });
    Ok(TORCH.get().expect("kept").bind(py))
}

pub(crate) fn is_tensor(obj: &Bound<'_, PyAny>) -> bool {
    let py = obj.py();
    if TENSOR.get().is_none() {
        let Ok(class) = pyffi::of_module(py, c"torch", c"Tensor") else {
            return false;
        };
        TENSOR.set(class.unbind());
    }
    obj.is_instance(TENSOR.get().expect("kept").bind(py)).unwrap_or(false)
}

fn info(obj: &Bound<'_, PyAny>) -> Option<TcfgTensorInfo> {
    let api = capi::api()?;
    let mut found = TcfgTensorInfo::empty();
    if unsafe { (api.tensor_info)(obj.as_ptr(), &mut found) } == 0 {
        return None;
    }
    Some(found)
}

fn version(obj: &Bound<'_, PyAny>) -> u64 {
    info(obj).map_or(capi::TCFG_NO_VERSION, |found| found.version)
}

/// Where the values of a tensor are. Another place behind the call says
/// that the tensor was given other data (`t.data = ...`), which moves no
/// version.
fn memory(obj: &Bound<'_, PyAny>) -> usize {
    info(obj).map_or(0, |found| found.data as usize)
}

fn requires_grad(obj: &Bound<'_, PyAny>) -> bool {
    info(obj).is_some_and(|found| found.requires_grad != 0)
}

/// What a tensor is a view of, if it is one: asked of the getter of the
/// class under every tensor, not by name on the tensor.
fn base_of<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    let base = unsafe { crate::graph::base_of(obj.as_ptr()) };
    if base.is_null() {
        return Err(PyRuntimeError::new_err("torch.vm: what a tensor is a view of cannot be read"));
    }
    let base = unsafe { Bound::from_owned_ptr(obj.py(), base) };
    Ok(if base.is_none() { None } else { Some(base) })
}

/// The node of autograd that made a tensor, if one did: asked of the
/// getter of the class under every tensor, not by name on the tensor.
fn node_of<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    let node = unsafe { crate::graph::node_of(obj.as_ptr()) };
    if node.is_null() {
        return Err(PyRuntimeError::new_err("torch.vm: the node that made a tensor cannot be read"));
    }
    let node = unsafe { Bound::from_owned_ptr(obj.py(), node) };
    Ok(if node.is_none() { None } else { Some(node) })
}

fn key(obj: &Bound<'_, PyAny>) -> usize {
    obj.as_ptr() as usize
}

/// `detach`, `requires_grad_` and `clone` of the class under every
/// tensor, found once: what is on a class of tensors under those names is
/// not asked.
fn tensor_methods(py: Python<'_>) -> PyResult<&'static (Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    static METHODS: Once<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> = Once::new();
    if METHODS.get().is_none() {
        let class = pyffi::of_module(py, c"torch._C", c"TensorBase")?;
        METHODS.set((class.getattr("detach")?.unbind(), class.getattr("requires_grad_")?.unbind(), class.getattr("clone")?.unbind()));
    }
    Ok(METHODS.get().expect("kept"))
}

/// A copy of a tensor, made by the method of the class under every
/// tensor.
fn copy<'py>(tensor: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    tensor_methods(tensor.py())?.2.bind(tensor.py()).call1((tensor,))
}

/// Another tensor in the memory of a tensor, with its version and without
/// its history: it holds the memory, not the tensor.
fn alias<'py>(tensor: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    tensor_methods(tensor.py())?.0.bind(tensor.py()).call1((tensor,))
}

/// An alias of a tensor that is a leaf of its own: where the forward runs
/// once more on the tape, the gradient at it is the contribution to the
/// gradient of the tensor.
fn leaf<'py>(tensor: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    tensor_methods(tensor.py())?.1.bind(tensor.py()).call1((alias(tensor)?,))
}

/// A tensor in the memory of a tensor, with its version, that has its
/// history and holds neither the tensor nor its Python object
/// (`torch.vm._handle.GraphInput`): it is no view of the tensor, and its
/// node hands a gradient on to the node that made the tensor.
fn handle<'py>(tensor: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    static MAKE: Once<Py<PyAny>> = Once::new();
    let py = tensor.py();
    if MAKE.get().is_none() {
        MAKE.set(pyffi::of_module(py, c"torch.vm._handle", c"GraphInput")?.getattr("apply")?.unbind());
    }
    served(5);
    MAKE.get().expect("kept").bind(py).call1((tensor,))
}

/// Whether the backward that runs on this thread retains the graph: asked
/// of the autograd engine. A backward that does not is followed by the
/// release of every node it ran.
fn retains(py: Python<'_>) -> PyResult<bool> {
    static ASK: Once<Py<PyAny>> = Once::new();
    if ASK.get().is_none() {
        ASK.set(torch(py)?.getattr("_C")?.getattr("_autograd")?.getattr("_get_current_graph_task_keep_graph")?.unbind());
    }
    ASK.get().expect("kept").bind(py).call0()?.extract()
}

/// How many backwards the backward graph of a form served, how many ran
/// the forward once more on the tape (without and with `create_graph`),
/// how many calls of a form kept eager's tape for an input whose storage
/// is watched, how many aliases of inputs the nodes were given, and how
/// many of those came with a tensor that has the history of the input.
static SERVED: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
const SERVED_NAMES: [&str; 6] = ["graph", "again", "create_graph", "watched", "aliases", "histories"];

fn served(which: usize) {
    SERVED[which].fetch_add(1, Ordering::Relaxed);
}

/// Whether a value is a device or a dtype.
fn kind_of(value: &Bound<'_, PyAny>) -> bool {
    let Ok(torch) = torch(value.py()) else {
        return false;
    };
    ["device", "dtype"].iter().any(|name| torch.getattr(*name).and_then(|class| value.is_instance(&class)).unwrap_or(false))
}

/// The tensors in a value: itself, or those of a tuple or list, in order.
fn tensors_in<'py>(value: &Bound<'py, PyAny>, into: &mut Vec<Bound<'py, PyAny>>) {
    if is_tensor(value) {
        into.push(value.clone());
    } else if value.is_instance_of::<PyTuple>() || value.is_instance_of::<PyList>() {
        if let Ok(items) = value.try_iter() {
            for item in items.flatten() {
                tensors_in(&item, into);
            }
        }
    }
}

/// A gradient to hand a root when its backward is taken down: not the same
/// number everywhere, and none drawn.
fn seed<'py>(py: Python<'py>, tensor: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let count: usize = tensor.call_method0("numel")?.extract()?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("device", tensor.getattr("device")?)?;
    let values = torch(py)?.call_method("arange", (count.max(1),), Some(&kwargs))?;
    let values = values.call_method1("__mod__", (7,))?.call_method1("__add__", (1,))?;
    let values = values.call_method1("to", (tensor.getattr("dtype")?,))?.call_method1("__truediv__", (8,))?;
    values.call_method1("reshape", (tensor.getattr("shape")?,))
}

fn same_bits(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<bool> {
    torch(a.py())?.call_method1("equal", (a, b))?.extract()
}

/// Sets the grad mode of the thread until it is dropped.
pub(crate) struct GradMode<'py> {
    torch: &'py Bound<'py, PyModule>,
    before: bool,
}

impl<'py> GradMode<'py> {
    pub(crate) fn set(py: Python<'py>, enabled: bool) -> PyResult<GradMode<'py>> {
        let torch = torch(py)?;
        let before: bool = torch.call_method0("is_grad_enabled")?.extract()?;
        torch.getattr("_C")?.call_method1("_set_grad_enabled", (enabled,))?;
        Ok(GradMode { torch, before })
    }
}

impl Drop for GradMode<'_> {
    fn drop(&mut self) {
        let _ = self.torch.getattr("_C").and_then(|c| c.call_method1("_set_grad_enabled", (self.before,)));
    }
}

/// Whether autocast is on for the thread, per kind of device, and to which
/// dtype it casts there.
struct Cast {
    on: [bool; 2],
    dtypes: [Py<PyAny>; 2],
}

const CAST_DEVICES: [&str; 2] = ["cpu", "cuda"];

impl Cast {
    fn now(py: Python<'_>) -> PyResult<Cast> {
        let torch = torch(py)?;
        let on = |device: &str| -> PyResult<bool> { torch.call_method1("is_autocast_enabled", (device,))?.extract() };
        let dtype = |device: &str| -> PyResult<Py<PyAny>> { Ok(torch.call_method1("get_autocast_dtype", (device,))?.unbind()) };
        Ok(Cast { on: [on("cpu")?, on("cuda")?], dtypes: [dtype("cpu")?, dtype("cuda")?] })
    }

    fn off(&self) -> bool {
        !self.on[0] && !self.on[1]
    }

    /// Whether a thread with this state casts as one with the other does.
    fn same(&self, other: &Cast) -> bool {
        (0..2).all(|at| self.on[at] == other.on[at] && (!self.on[at] || self.dtypes[at].is(&other.dtypes[at])))
    }

    /// Sets the thread to this state, or with `off` to no autocast, until
    /// what is returned is dropped.
    fn set<'py>(&self, py: Python<'py>, off: bool) -> PyResult<CastGuard<'py>> {
        let before = Cast::now(py)?;
        let torch = torch(py)?;
        for (at, device) in CAST_DEVICES.iter().enumerate() {
            torch.call_method1("set_autocast_enabled", (*device, self.on[at] && !off))?;
            if self.on[at] && !off {
                torch.call_method1("set_autocast_dtype", (*device, self.dtypes[at].bind(py)))?;
            }
        }
        Ok(CastGuard { torch, before })
    }
}

struct CastGuard<'py> {
    torch: &'py Bound<'py, PyModule>,
    before: Cast,
}

impl Drop for CastGuard<'_> {
    fn drop(&mut self) {
        let py = self.torch.py();
        for (at, device) in CAST_DEVICES.iter().enumerate() {
            let _ = self.torch.call_method1("set_autocast_enabled", (*device, self.before.on[at]));
            let _ = self.torch.call_method1("set_autocast_dtype", (*device, self.before.dtypes[at].bind(py)));
        }
    }
}

/// How many backwards are being taken down with cuDNN's choice of
/// algorithms set aside, and what its two settings were.
static STEADY: Mutex<(u32, bool, bool)> = Mutex::new((0, false, false));

/// While it lives, cuDNN chooses its algorithms the same way on every
/// thread: its benchmark mode is off and its deterministic mode is on. In
/// benchmark mode cuDNN picks the algorithm of a convolution by what it
/// measured, and remembers what it picked, per thread: what a node gives on
/// the thread that takes a form down is then not, bit for bit, what the
/// autograd engine's thread gives for the same tape, though both are
/// right. The settings are the process's, so they are set for all threads
/// while a backward is taken down and compared; the kernels that the two
/// graphs run afterwards choose as the caller set it, on the threads
/// eager's would run on.
struct Steady<'py> {
    c: Bound<'py, PyAny>,
}

impl<'py> Steady<'py> {
    fn hold(py: Python<'py>) -> PyResult<Steady<'py>> {
        let c = torch(py)?.getattr("_C")?;
        let mut held = STEADY.lock().unwrap();
        if held.0 == 0 {
            held.1 = c.call_method0("_get_cudnn_benchmark")?.extract()?;
            held.2 = c.call_method0("_get_cudnn_deterministic")?.extract()?;
            c.call_method1("_set_cudnn_benchmark", (false,))?;
            c.call_method1("_set_cudnn_deterministic", (true,))?;
        }
        held.0 += 1;
        Ok(Steady { c })
    }
}

impl Drop for Steady<'_> {
    fn drop(&mut self) {
        let mut held = STEADY.lock().unwrap();
        held.0 -= 1;
        if held.0 == 0 {
            let _ = self.c.call_method1("_set_cudnn_benchmark", (held.1,));
            let _ = self.c.call_method1("_set_cudnn_deterministic", (held.2,));
        }
    }
}

/// Why this call runs on eager's tape, if something is on that the engine's
/// backward does not serve: looked at at every call of a form.
fn needs_tape(py: Python<'_>) -> PyResult<Option<&'static str>> {
    let torch = torch(py)?;
    if torch.call_method0("is_anomaly_enabled")?.extract()? {
        return Ok(Some("anomaly mode"));
    }
    let c = torch.getattr("_C")?;
    if !c.getattr("_functorch")?.call_method0("peek_interpreter_stack")?.is_none() {
        return Ok(Some("a transform of torch.func"));
    }
    let level: i64 = torch.getattr("autograd")?.getattr("forward_ad")?.getattr("_current_level")?.extract()?;
    if level >= 0 {
        return Ok(Some("a forward-mode level"));
    }
    if !c.getattr("_autograd")?.call_method1("_top_saved_tensors_default_hooks", (true,))?.is_none() {
        return Ok(Some("saved-tensor hooks"));
    }
    Ok(None)
}

/// What the autograd engine gives for `inputs` when `seeds` come in at
/// `roots`, the graph kept and inputs that are not reached given as None:
/// `torch.autograd.grad(roots, inputs, seeds, retain_graph=True,
/// allow_unused=True)` without the Python code around the engine, which
/// imports at every call (two statements per root) and so calls
/// `builtins.__import__` where eager calls nothing. The context is stashed
/// for the engine's device threads as that code stashes it.
fn engine_grad<'py>(py: Python<'py>, roots: Bound<'py, PyTuple>, inputs: Bound<'py, PyTuple>, seeds: Bound<'py, PyTuple>) -> PyResult<Bound<'py, PyAny>> {
    let engine = pyffi::of_module(py, c"torch.autograd.variable", c"Variable")?.getattr("_execution_engine")?;
    let c = pyffi::of_module(py, c"torch", c"_C")?;
    c.call_method1("_stash_obj_in_tls", ("context", pyffi::of_module(py, c"contextvars", c"copy_context")?.call0()?))?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("accumulate_grad", false)?;
    let given = engine.call_method("run_backward", (roots, seeds, true, false, inputs, true), Some(&kwargs));
    c.call_method1("_remove_obj_from_tls", ("context",))?;
    given
}

// ---------------------------------------------------------------------------
// The C API, as far as a backward is derived with it.

fn bapi() -> PyResult<&'static TcfgBackwardApi> {
    capi::backward_api().ok_or_else(|| PyRuntimeError::new_err("torch.vm: the runtime was not handed the C API of torch._C"))
}

/// One kernel a tape saw.
struct Record<'py> {
    op: Bound<'py, PyAny>,
    name: String,
    args: Bound<'py, PyTuple>,
    out: Bound<'py, PyAny>,
    written: Vec<usize>,
    random: bool,
    view: bool,
}

/// Operators that say something of a tensor without looking at what it
/// holds: no operation of a graph.
const ASKS: [&str; 16] = [
    "sym_size", "size", "dim", "sym_numel", "numel", "sym_stride", "stride", "sym_storage_offset", "storage_offset", "is_same_size",
    "is_floating_point", "is_complex", "is_signed", "is_inference", "is_conj", "is_neg",
];

impl Record<'_> {
    fn base(&self) -> &str {
        self.name.split("::").last().unwrap_or("").split('.').next().unwrap_or("")
    }

    /// A kernel that only says something of a tensor's sizes.
    fn asks(&self) -> bool {
        let mut found = Vec::new();
        tensors_in(&self.out, &mut found);
        found.is_empty() && ASKS.contains(&self.base())
    }
}

struct Tape;

impl Tape {
    fn begin() -> PyResult<Tape> {
        unsafe { (bapi()?.tape_begin)() };
        Ok(Tape)
    }

    fn end<'py>(self, py: Python<'py>) -> PyResult<Vec<Record<'py>>> {
        let api = bapi()?;
        std::mem::forget(self);
        let list = unsafe { Bound::from_owned_ptr_or_err(py, (api.tape_end)())? };
        let mut records = Vec::new();
        for item in list.try_iter()? {
            let item = item?;
            let record = Record {
                op: item.get_item(0)?,
                name: item.get_item(1)?.extract()?,
                args: item.get_item(2)?.cast_into::<PyTuple>().map_err(|_| PyTypeError::new_err("the arguments of a kernel are a tuple"))?,
                out: item.get_item(3)?,
                written: item.get_item(4)?.extract()?,
                random: item.get_item(5)?.extract()?,
                view: item.get_item(6)?.extract()?,
            };
            if !record.asks() {
                records.push(record);
            }
        }
        Ok(records)
    }
}

impl Drop for Tape {
    // An operation raised under the tape: what it saw is let go.
    fn drop(&mut self) {
        if let Some(api) = capi::backward_api() {
            let list = unsafe { (api.tape_end)() };
            if list.is_null() {
                unsafe { pyo3::ffi::PyErr_Clear() };
            } else {
                unsafe { pyo3::ffi::Py_DECREF(list) };
            }
        }
    }
}

/// A node of the autograd graph, as the engine reads it.
#[derive(Clone)]
struct Node<'py> {
    obj: Bound<'py, PyAny>,
    name: String,
    sequence: u64,
    inputs: u32,
    edges: u32,
    flags: u32,
}

impl<'py> Node<'py> {
    fn of(obj: Bound<'py, PyAny>) -> PyResult<Node<'py>> {
        let mut found = TcfgNodeInfo { name: std::ptr::null(), sequence_nr: 0, n_inputs: 0, n_edges: 0, flags: 0 };
        if unsafe { (bapi()?.node_info)(obj.as_ptr(), &mut found) } == 0 {
            return Err(PyTypeError::new_err("what a gradient goes to is no node of autograd"));
        }
        let name = if found.flags & TCFG_NODE_PYTHON != 0 {
            obj.get_type().name()?.to_string()
        } else {
            unsafe { CStr::from_ptr(found.name) }.to_string_lossy().into_owned()
        };
        Ok(Node { obj, name, sequence: found.sequence_nr, inputs: found.n_inputs, edges: found.n_edges, flags: found.flags })
    }

    /// The node the gradient of a tensor goes to, and which of its inputs
    /// the tensor is; None for a tensor that needs no gradient.
    fn of_tensor(tensor: &Bound<'py, PyAny>) -> PyResult<Option<(Node<'py>, u32)>> {
        let py = tensor.py();
        let mut input = 0u32;
        let found = unsafe { (bapi()?.node_of)(tensor.as_ptr(), &mut input) };
        if found.is_null() {
            return PyErr::take(py).map_or(Ok(None), Err);
        }
        Ok(Some((Node::of(unsafe { Bound::from_owned_ptr(py, found) })?, input)))
    }

    /// Where an edge leads: the node, and which of its inputs.
    fn next(&self, edge: u32) -> PyResult<Option<(Node<'py>, u32)>> {
        let py = self.obj.py();
        let mut input = 0u32;
        let found = unsafe { (bapi()?.node_next)(self.obj.as_ptr(), edge, &mut input) };
        if found.is_null() {
            return PyErr::take(py).map_or(Ok(None), Err);
        }
        Ok(Some((Node::of(unsafe { Bound::from_owned_ptr(py, found) })?, input)))
    }

    /// Calls the node as the autograd engine calls it.
    fn call(&self, grads: &[Option<Bound<'py, PyAny>>], twice: bool) -> PyResult<Vec<Option<Bound<'py, PyAny>>>> {
        let py = self.obj.py();
        let handed = PyTuple::new(py, grads.iter().map(|grad| grad.clone().unwrap_or_else(|| py.None().into_bound(py))))?;
        let flags = if twice { TCFG_RUN_ON_TAPE } else { 0 };
        let given = unsafe { Bound::from_owned_ptr_or_err(py, (bapi()?.node_call)(self.obj.as_ptr(), handed.as_ptr(), flags))? };
        Ok(given.try_iter()?.flatten().map(|item| if item.is_none() { None } else { Some(item) }).collect())
    }

    fn key(&self) -> usize {
        key(&self.obj)
    }
}

/// Calls an operator with the arguments of its schema, in order.
fn op_call<'py>(op: &Bound<'py, PyAny>, args: &Bound<'py, PyTuple>, flags: u32) -> PyResult<Bound<'py, PyAny>> {
    unsafe { Bound::from_owned_ptr_or_err(op.py(), (bapi()?.op_call)(op.as_ptr(), args.as_ptr(), flags)) }
}

/// A straight-line program of operators over the values of a joint graph.
struct Program {
    raw: *mut c_void,
    takes: Vec<Value>,
    gives: Vec<Value>,
}

// A program is only run and let go with the GIL held.
unsafe impl Send for Program {}
unsafe impl Sync for Program {}

impl Program {
    fn new(ops: &Bound<'_, PyAny>, values: u32, takes: Vec<Value>, gives: Vec<Value>) -> Result<Program, Stop> {
        let mut reason = [0 as c_char; 256];
        let raw = unsafe { (bapi()?.program_new)(ops.as_ptr(), values, reason.as_mut_ptr(), reason.len()) };
        if raw.is_null() {
            let why = unsafe { CStr::from_ptr(reason.as_ptr()) }.to_string_lossy().into_owned();
            return refused(format!("an operation is none the engine runs: {why}"));
        }
        Ok(Program { raw, takes, gives })
    }

    fn run<'py>(&self, py: Python<'py>, inputs: &[Bound<'py, PyAny>], flags: u32) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let handed: Vec<Obj> = inputs.iter().map(|input| input.as_ptr()).collect();
        let given = unsafe {
            let run = bapi()?.program_run;
            let raw = run(self.raw, handed.as_ptr(), self.takes.as_ptr(), self.takes.len() as u32, self.gives.as_ptr(), self.gives.len() as u32, flags);
            Bound::from_owned_ptr_or_err(py, raw)?
        };
        Ok(given.try_iter()?.flatten().collect())
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        if let Some(api) = capi::backward_api() {
            unsafe { (api.program_free)(self.raw) };
        }
    }
}

/// What runs one of the two graphs of a form: the engine's program, or what
/// the passes over graphs made of it.
enum Runner {
    Program(Program),
    Made(Py<PyAny>, usize),
}

impl Runner {
    fn run<'py>(&self, py: Python<'py>, inputs: &[Bound<'py, PyAny>]) -> PyResult<Vec<Bound<'py, PyAny>>> {
        match self {
            Runner::Program(program) => program.run(py, inputs, TCFG_RUN_BELOW_AUTOGRAD),
            Runner::Made(callable, count) => {
                // A pass is handed tensors that need no gradient, and what
                // it made runs without one: nothing is left to derive.
                let _mode = GradMode::set(py, false)?;
                let given = callable.bind(py).call1(PyTuple::new(py, inputs)?)?;
                let given: Vec<Bound<'py, PyAny>> = given.try_iter()?.flatten().collect();
                if given.len() != *count {
                    return Err(PyRuntimeError::new_err("what a pass made of a graph of the engine's backward does not give its values"));
                }
                Ok(given)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The graph a form is: the operations the engine describes for the exporter
// (`Path::describe` in graph.rs), or, for a part of a form that was cut, the
// kernels of the part.

enum Operand {
    Value(u32),
    Several(Vec<Operand>, bool),
    Const(Py<PyAny>),
}

/// An argument of a kernel call, as the call site has it.
enum SiteArg {
    Operand(usize, bool),
    Packed(Vec<usize>),
    Const(Py<PyAny>),
}

enum Kind {
    /// A kernel call: the operator overload, its arguments by name (the
    /// keyword-only ones named), and the operands of the call site.
    Aten(Py<PyAny>, Vec<(Option<String>, SiteArg)>, Vec<Operand>),
    /// A call of a function, or with the flag of the method of that name of
    /// the first operand; the last operands by the names.
    Call(Option<Operand>, Option<String>, Vec<Operand>, Vec<String>),
    GetItem(Operand, Operand),
    SetItem(Operand, Operand, Operand),
    Same(Operand),
    Item(Operand, Operand),
    Attr(Operand, String),
    Size(Operand, i64),
    Dim(Operand),
    Binary(String, Operand, Operand),
    Neg(Operand),
    Tuple(Vec<Operand>),
    List(Vec<Operand>),
    Slice(Operand, Operand, Operand),
    /// A kernel of the joint graph of the form this is a part of: the
    /// operator, its arguments in the order of the schema, and where what
    /// it gives goes.
    Kernel(Py<PyAny>, Vec<Operand>, Vec<(String, Operand)>, Vec<(u32, u32)>, bool),
}

struct Step {
    kind: Kind,
    value: Option<u32>,
}

struct Graph {
    inputs: Vec<u32>,
    steps: Vec<Step>,
    outputs: Vec<u32>,
    /// Per output: whether something can hold it when the call that the
    /// graph ran for goes on.
    held: Vec<bool>,
    /// The ranks and sizes the engine checks of tensors it does not get
    /// back: the tensor, and the dimension for a size.
    exits: Vec<(Operand, Option<i64>)>,
    values: usize,
    /// For a part of a form that is cut: whether the results of that form
    /// were looked at together for a second backward (`returns_twice`),
    /// so that the part is not looked at by itself.
    checked: bool,
}

fn operand(described: &Bound<'_, PyAny>) -> PyResult<Operand> {
    let kind: String = described.get_item(0)?.extract()?;
    let value = described.get_item(1)?;
    Ok(match kind.as_str() {
        "v" => Operand::Value(value.extract()?),
        "l" | "t" => Operand::Several(value.try_iter()?.map(|item| operand(&item?)).collect::<PyResult<_>>()?, kind == "t"),
        _ => Operand::Const(value.unbind()),
    })
}

fn operands(described: &Bound<'_, PyAny>) -> PyResult<Vec<Operand>> {
    described.try_iter()?.map(|item| operand(&item?)).collect()
}

fn names(described: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if described.is_none() {
        return Ok(Vec::new());
    }
    described.try_iter()?.map(|item| item?.extract()).collect()
}

impl Operand {
    fn reads(&self, into: &mut Vec<u32>) {
        match self {
            Operand::Value(value) => into.push(*value),
            Operand::Several(items, _) => items.iter().for_each(|item| item.reads(into)),
            Operand::Const(_) => {}
        }
    }

    fn get<'py>(&self, py: Python<'py>, values: &[Option<Py<PyAny>>]) -> PyResult<Bound<'py, PyAny>> {
        Ok(match self {
            Operand::Value(value) => match values.get(*value as usize).and_then(|held| held.as_ref()) {
                Some(held) => held.bind(py).clone(),
                None => return Err(PyRuntimeError::new_err("torch.vm: a value of a graph is read before it is there")),
            },
            Operand::Several(items, tuple) => {
                let found: Vec<Bound<'py, PyAny>> = items.iter().map(|item| item.get(py, values)).collect::<PyResult<_>>()?;
                if *tuple {
                    PyTuple::new(py, found)?.into_any()
                } else {
                    PyList::new(py, found)?.into_any()
                }
            }
            Operand::Const(held) => held.bind(py).clone(),
        })
    }
}

impl Step {
    fn reads(&self) -> Vec<u32> {
        let mut found = Vec::new();
        let mut all = |items: &[&Operand]| items.iter().for_each(|item| item.reads(&mut found));
        match &self.kind {
            Kind::Aten(_, _, taken) | Kind::Tuple(taken) | Kind::List(taken) => all(&taken.iter().collect::<Vec<_>>()),
            Kind::Call(callee, _, taken, _) => {
                all(&callee.iter().collect::<Vec<_>>());
                all(&taken.iter().collect::<Vec<_>>());
            }
            Kind::GetItem(a, b) | Kind::Item(a, b) | Kind::Binary(_, a, b) => all(&[a, b]),
            Kind::SetItem(a, b, c) | Kind::Slice(a, b, c) => all(&[a, b, c]),
            Kind::Same(a) | Kind::Attr(a, _) | Kind::Size(a, _) | Kind::Dim(a) | Kind::Neg(a) => all(&[a]),
            Kind::Kernel(_, args, kwargs, _, _) => {
                all(&args.iter().collect::<Vec<_>>());
                all(&kwargs.iter().map(|(_, arg)| arg).collect::<Vec<_>>());
            }
        }
        found
    }

    /// The values the step gives.
    fn gives(&self) -> Vec<u32> {
        match &self.kind {
            Kind::Kernel(_, _, _, places, _) => places.iter().map(|(_, value)| *value).collect(),
            _ => self.value.into_iter().collect(),
        }
    }

    /// Whether the step is one that cannot run a kernel.
    fn plain(&self) -> bool {
        !matches!(self.kind, Kind::Aten(..) | Kind::Call(..) | Kind::GetItem(..) | Kind::SetItem(..) | Kind::Kernel(..))
    }

    /// The function a call calls, and what it is called with: for a call
    /// of the form's graph, and for an operation of a joint graph that is
    /// a call and no operator.
    #[allow(clippy::type_complexity)]
    fn called<'py>(&self, py: Python<'py>, values: &[Option<Py<PyAny>>]) -> PyResult<Option<(Bound<'py, PyAny>, Bound<'py, PyTuple>, Bound<'py, PyDict>)>> {
        if let Kind::Kernel(op, args, kwargs, _, _) = &self.kind {
            if op.bind(py).is_instance_of::<PyCapsule>() {
                return Ok(None);
            }
            let given: Vec<Bound<'py, PyAny>> = args.iter().map(|item| item.get(py, values)).collect::<PyResult<_>>()?;
            let named = PyDict::new(py);
            for (name, arg) in kwargs {
                named.set_item(name, arg.get(py, values)?)?;
            }
            return Ok(Some((op.bind(py).clone(), PyTuple::new(py, given)?, named)));
        }
        let Kind::Call(callee, method, taken, kwnames) = &self.kind else {
            return Ok(None);
        };
        let mut given: Vec<Bound<'py, PyAny>> = taken.iter().map(|item| item.get(py, values)).collect::<PyResult<_>>()?;
        let named = given.split_off(given.len() - kwnames.len());
        let kwargs = PyDict::new(py);
        for (name, value) in kwnames.iter().zip(named) {
            kwargs.set_item(name, value)?;
        }
        let function = match (callee, method) {
            (Some(callee), _) => callee.get(py, values)?,
            (None, Some(name)) => given.remove(0).getattr(name.as_str())?,
            (None, None) => return Err(PyRuntimeError::new_err("torch.vm: a call of nothing")),
        };
        Ok(Some((function, PyTuple::new(py, given)?, kwargs)))
    }

    /// Runs the step as the thread is, and puts what it gives among the
    /// values.
    fn run(&self, py: Python<'_>, values: &mut [Option<Py<PyAny>>]) -> PyResult<()> {
        let get = |item: &Operand| item.get(py, values);
        let given = match &self.kind {
            Kind::Aten(target, args, taken) => {
                let mut positional = Vec::new();
                let kwargs = PyDict::new(py);
                for (name, arg) in args {
                    let value = match arg {
                        SiteArg::Operand(at, listed) => {
                            let value = get(&taken[*at])?;
                            if *listed {
                                PyList::new(py, [value])?.into_any()
                            } else {
                                value
                            }
                        }
                        SiteArg::Packed(which) => PyList::new(py, which.iter().map(|at| get(&taken[*at])).collect::<PyResult<Vec<_>>>()?)?.into_any(),
                        SiteArg::Const(held) => held.bind(py).clone(),
                    };
                    match name {
                        Some(name) => kwargs.set_item(name, value)?,
                        None => positional.push(value),
                    }
                }
                target.bind(py).call(PyTuple::new(py, positional)?, Some(&kwargs))?
            }
            Kind::Call(..) => {
                let (function, args, kwargs) = self.called(py, values)?.expect("a call");
                function.call(args, Some(&kwargs))?
            }
            Kind::GetItem(a, b) | Kind::Item(a, b) => get(a)?.get_item(get(b)?)?,
            Kind::SetItem(a, b, c) => {
                get(a)?.set_item(get(b)?, get(c)?)?;
                return Ok(());
            }
            Kind::Same(a) => get(a)?,
            Kind::Attr(a, name) => get(a)?.getattr(name.as_str())?,
            Kind::Size(a, dim) => get(a)?.call_method1("size", (*dim,))?,
            Kind::Dim(a) => get(a)?.call_method0("dim")?,
            Kind::Binary(symbol, a, b) => {
                let method = match symbol.as_str() {
                    "+" => "__add__",
                    "-" => "__sub__",
                    "*" => "__mul__",
                    "//" => "__floordiv__",
                    "%" => "__mod__",
                    _ => return Err(PyRuntimeError::new_err(format!("torch.vm: the engine describes no operation {symbol}"))),
                };
                get(a)?.call_method1(method, (get(b)?,))?
            }
            Kind::Neg(a) => get(a)?.call_method0("__neg__")?,
            Kind::Tuple(items) => PyTuple::new(py, items.iter().map(get).collect::<PyResult<Vec<_>>>()?)?.into_any(),
            Kind::List(items) => PyList::new(py, items.iter().map(get).collect::<PyResult<Vec<_>>>()?)?.into_any(),
            Kind::Slice(a, b, c) => {
                let (a, b, c) = (get(a)?, get(b)?, get(c)?);
                unsafe { Bound::from_owned_ptr_or_err(py, pyo3::ffi::PySlice_New(a.as_ptr(), b.as_ptr(), c.as_ptr()))? }
            }
            Kind::Kernel(op, args, kwargs, places, single) => {
                let args = PyTuple::new(py, args.iter().map(get).collect::<PyResult<Vec<_>>>()?)?;
                let given = if op.bind(py).is_instance_of::<PyCapsule>() {
                    op_call(op.bind(py), &args, 0)?
                } else {
                    let named = PyDict::new(py);
                    for (name, arg) in kwargs {
                        named.set_item(name, get(arg)?)?;
                    }
                    op.bind(py).call(args, Some(&named))?
                };
                for (place, value) in places {
                    let item = if *single { given.clone() } else { given.get_item(*place)? };
                    values[*value as usize] = Some(item.unbind());
                }
                return Ok(());
            }
        };
        if let Some(value) = self.value {
            values[value as usize] = Some(given.unbind());
        }
        Ok(())
    }
}

/// The operator a call site calls, and its arguments as the site has them:
/// by name for those the operator takes by name only.
#[allow(clippy::type_complexity)]
fn site_of<'py>(py: Python<'py>, site: &Bound<'py, PyAny>) -> PyResult<(Bound<'py, PyAny>, Vec<(Option<String>, SiteArg)>)> {
    let (name, overload): (String, String) = (site.get_item("op")?.extract()?, site.get_item("overload")?.extract()?);
    let (namespace, op) = name.split_once("::").unwrap_or(("aten", &name));
    let overload = if overload.is_empty() { "default" } else { &overload };
    let target = torch(py)?.getattr("ops")?.getattr(namespace)?.getattr(op)?.getattr(overload)?;
    let mut args = Vec::new();
    for arg in site.get_item("args")?.try_iter()? {
        let arg = arg?;
        let (name, keyword_only, source): (String, bool, String) = (arg.get_item(0)?.extract()?, arg.get_item(1)?.extract()?, arg.get_item(2)?.extract()?);
        let payload = arg.get_item(3)?;
        let found = match source.as_str() {
            "operand" => SiteArg::Operand(payload.get_item(0)?.extract()?, payload.get_item(1)?.extract::<String>().is_ok_and(|how| how == "int_as_list")),
            "packed" => SiteArg::Packed(payload.extract()?),
            _ => SiteArg::Const(payload.unbind()),
        };
        args.push((keyword_only.then_some(name), found));
    }
    Ok((target, args))
}

/// Calls the operator of a call site, as `site_export` describes one, with
/// `operand(k)` for the k-th value the site takes.
pub(crate) fn site_call<'py>(py: Python<'py>, site: &Bound<'py, PyAny>, operand: &dyn Fn(usize) -> PyResult<Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    let (target, args) = site_of(py, site)?;
    let mut positional = Vec::new();
    let kwargs = PyDict::new(py);
    for (name, arg) in &args {
        let value = match arg {
            SiteArg::Operand(at, listed) => {
                let value = operand(*at)?;
                if *listed {
                    PyList::new(py, [value])?.into_any()
                } else {
                    value
                }
            }
            SiteArg::Packed(which) => PyList::new(py, which.iter().map(|at| operand(*at)).collect::<PyResult<Vec<_>>>()?)?.into_any(),
            SiteArg::Const(held) => held.bind(py).clone(),
        };
        match name {
            Some(name) => kwargs.set_item(name, value)?,
            None => positional.push(value),
        }
    }
    target.call(PyTuple::new(py, positional)?, Some(&kwargs))
}

impl Graph {
    /// The graph the engine describes for the exporter.
    fn described(py: Python<'_>, description: &Bound<'_, PyDict>) -> PyResult<Graph> {
        let need = |name: &str| description.get_item(name)?.ok_or_else(|| PyValueError::new_err(format!("the description of a graph has no {name}")));
        let mut steps = Vec::new();
        for node in need("nodes")?.try_iter()? {
            let node = node?;
            let kind: String = node.get_item(0)?.extract()?;
            let part = |at: usize| node.get_item(at);
            let value = |at: usize| -> PyResult<Option<u32>> { Ok(Some(part(at)?.extract()?)) };
            let step = match kind.as_str() {
                "aten" => {
                    let (target, args) = site_of(py, &part(1)?)?;
                    Step { kind: Kind::Aten(target.unbind(), args, operands(&part(2)?)?), value: value(3)? }
                }
                "call" => Step { kind: Kind::Call(Some(operand(&part(1)?)?), None, operands(&part(2)?)?, names(&part(3)?)?), value: value(4)? },
                "method" => Step { kind: Kind::Call(None, Some(part(1)?.extract()?), operands(&part(2)?)?, names(&part(3)?)?), value: value(4)? },
                "getitem" => {
                    let taken = part(2)?;
                    Step { kind: Kind::GetItem(operand(&taken.get_item(0)?)?, operand(&taken.get_item(1)?)?), value: value(4)? }
                }
                "setitem" => {
                    let taken = part(2)?;
                    Step { kind: Kind::SetItem(operand(&taken.get_item(0)?)?, operand(&taken.get_item(1)?)?, operand(&taken.get_item(2)?)?), value: None }
                }
                "same" => Step { kind: Kind::Same(operand(&part(1)?)?), value: value(2)? },
                "item" => Step { kind: Kind::Item(operand(&part(1)?)?, operand(&part(2)?)?), value: value(3)? },
                "attr" => Step { kind: Kind::Attr(operand(&part(1)?)?, part(2)?.extract()?), value: value(3)? },
                "size" => Step { kind: Kind::Size(operand(&part(1)?)?, part(2)?.extract()?), value: value(3)? },
                "dim" => Step { kind: Kind::Dim(operand(&part(1)?)?), value: value(2)? },
                "binary" => Step { kind: Kind::Binary(part(1)?.extract()?, operand(&part(2)?)?, operand(&part(3)?)?), value: value(4)? },
                "neg" => Step { kind: Kind::Neg(operand(&part(1)?)?), value: value(2)? },
                "tuple" => Step { kind: Kind::Tuple(operands(&part(1)?)?), value: value(2)? },
                "list" => Step { kind: Kind::List(operands(&part(1)?)?), value: value(2)? },
                "slice" => {
                    let taken = part(1)?;
                    let of = |at: usize| operand(&taken.get_item(at)?);
                    Step { kind: Kind::Slice(of(0)?, of(1)?, of(2)?), value: value(2)? }
                }
                other => return Err(PyValueError::new_err(format!("the engine describes no node of kind {other}"))),
            };
            steps.push(step);
        }
        let inputs: Vec<u32> = need("input_ids")?.extract()?;
        let outputs: Vec<u32> = need("outputs")?.extract()?;
        let held: Vec<bool> = match description.get_item("held")? {
            Some(held) => held.extract()?,
            None => vec![true; outputs.len()],
        };
        if held.len() != outputs.len() {
            return Err(PyValueError::new_err("the description of a graph says of other values than its outputs whether they are held"));
        }
        let mut exits = Vec::new();
        for exit in need("exits")?.try_iter()? {
            let exit = exit?;
            let kind: String = exit.get_item(0)?.extract()?;
            let dim = if kind == "size" { Some(exit.get_item(2)?.extract()?) } else { None };
            exits.push((operand(&exit.get_item(1)?)?, dim));
        }
        let mut graph = Graph { inputs, steps, outputs, held, exits, values: 0, checked: false };
        graph.count();
        Ok(graph)
    }

    /// Operations of a joint graph as a graph of their own: it takes the
    /// values `takes` and gives the values `gives`, of which `held` says
    /// which something can hold behind the form.
    fn kernels(ops: &Bound<'_, PyList>, takes: Vec<Value>, gives: Vec<Value>, held: Vec<bool>, checked: bool) -> PyResult<Graph> {
        let mut steps = Vec::new();
        for op in ops.iter() {
            let mut kwargs = Vec::new();
            for (name, arg) in op.get_item(2)?.cast::<PyDict>().map_err(|_| PyTypeError::new_err("the keyword arguments of an operation are a dict"))?.iter() {
                kwargs.push((name.extract()?, operand(&arg)?));
            }
            let places: Vec<(u32, u32)> = op.get_item(3)?.extract()?;
            let kind = Kind::Kernel(op.get_item(0)?.unbind(), operands(&op.get_item(1)?)?, kwargs, places, op.get_item(4)?.extract()?);
            steps.push(Step { kind, value: None });
        }
        let mut graph = Graph { inputs: takes, steps, outputs: gives, held, exits: Vec::new(), values: 0, checked };
        graph.count();
        Ok(graph)
    }

    fn count(&mut self) {
        let mut most = self.inputs.iter().chain(&self.outputs).copied().max().unwrap_or(0);
        for step in &self.steps {
            most = step.gives().into_iter().chain(step.reads()).fold(most, u32::max);
        }
        self.values = most as usize + 1;
    }

    fn table<'py>(&self, feeds: &[Bound<'py, PyAny>]) -> PyResult<Vec<Option<Py<PyAny>>>> {
        if feeds.len() != self.inputs.len() {
            return Err(PyTypeError::new_err(format!("a graph of {} inputs is called with {}", self.inputs.len(), feeds.len())));
        }
        let mut values: Vec<Option<Py<PyAny>>> = (0..self.values).map(|_| None).collect();
        for (value, feed) in self.inputs.iter().zip(feeds) {
            values[*value as usize] = Some(feed.clone().unbind());
        }
        Ok(values)
    }

    /// What the graph gives, of its values: its outputs, and the ranks and
    /// sizes the engine checks.
    fn given<'py>(&self, py: Python<'py>, values: &[Option<Py<PyAny>>]) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let mut given = Vec::new();
        for value in &self.outputs {
            given.push(Operand::Value(*value).get(py, values)?);
        }
        for (tensor, dim) in &self.exits {
            let tensor = tensor.get(py, values)?;
            given.push(match dim {
                Some(dim) => tensor.call_method1("size", (*dim,))?,
                None => tensor.call_method0("dim")?,
            });
        }
        Ok(given)
    }

    /// Runs the graph as the thread is: on eager's tape, where gradients
    /// are enabled.
    fn run<'py>(&self, py: Python<'py>, feeds: &[Bound<'py, PyAny>]) -> PyResult<Vec<Option<Py<PyAny>>>> {
        let mut values = self.table(feeds)?;
        for step in &self.steps {
            step.run(py, &mut values)?;
        }
        Ok(values)
    }

    /// What is computed behind the kernels: the steps that run none (`ran`
    /// says which did) and that what the graph gives needs, and the values
    /// they take of those that ran kernels.
    fn behind(&self, ran: &[bool]) -> (Vec<usize>, Vec<u32>) {
        let mut wanted: HashSet<u32> = self.outputs.iter().copied().collect();
        for (tensor, _) in &self.exits {
            let mut found = Vec::new();
            tensor.reads(&mut found);
            wanted.extend(found);
        }
        let (mut steps, mut taken) = (Vec::new(), Vec::new());
        for (at, step) in self.steps.iter().enumerate().rev() {
            let gives = step.gives();
            if !gives.iter().any(|value| wanted.contains(value)) {
                continue;
            }
            if ran[at] {
                taken.extend(gives.into_iter().filter(|value| wanted.contains(value)));
            } else {
                steps.push(at);
                wanted.extend(step.reads());
            }
        }
        steps.reverse();
        taken.sort_unstable();
        taken.dedup();
        (steps, taken)
    }
}

// ---------------------------------------------------------------------------
// Eager's nodes for one run of a form.

/// The roots, the nodes between the roots and the inputs in the order the
/// autograd engine runs them, and the contributions they make to the
/// gradients of the inputs, numbered in that order.
struct Nodes<'py> {
    roots: Vec<Bound<'py, PyAny>>,
    /// Where the gradient of each root comes in: its node, and which input.
    seats: Vec<(usize, u32)>,
    /// Where the gradients of the inputs go: (node, input) to the input.
    edges: HashMap<(usize, u32), u32>,
    /// The inputs another input was computed of. What the autograd engine
    /// gives for one of those holds what comes to it through the other as
    /// well, and to give it the engine runs nodes that are not of this
    /// graph: it is not asked for them.
    upstream: HashSet<u32>,
    order: Vec<Node<'py>>,
    /// Per node of `order`, per edge: where it leads.
    next: Vec<Vec<Option<(usize, u32)>>>,
    /// (node of `order`, edge) to which contribution it is.
    slots: HashMap<(usize, u32), u32>,
    /// Per contribution: the input it is for.
    targets: Vec<u32>,
    /// The other nodes that were looked at: one that died could give its
    /// place to another.
    _held: Vec<Node<'py>>,
}

impl<'py> Nodes<'py> {
    fn new(feeds: &[Bound<'py, PyAny>], roots: Vec<Bound<'py, PyAny>>) -> Result<Nodes<'py>, Stop> {
        Nodes::over(feeds.iter().map(Node::of_tensor).collect::<PyResult<_>>()?, roots)
    }

    /// With, per input, the node its gradient goes to and which of that
    /// node's inputs it is, in place of the input.
    fn over(leads: Vec<Option<(Node<'py>, u32)>>, roots: Vec<Bound<'py, PyAny>>) -> Result<Nodes<'py>, Stop> {
        let mut edges: HashMap<(usize, u32), u32> = HashMap::new();
        let mut starts = Vec::new();
        for (at, lead) in leads.into_iter().enumerate() {
            if let Some((node, input)) = lead {
                edges.entry((node.key(), input)).or_insert(at as u32);
                starts.push(node);
            }
        }
        // The nodes that are looked at are held: one that died could give
        // its place to another.
        let mut held: HashMap<usize, Node<'py>> = HashMap::new();
        let mut upstream = HashSet::new();
        let mut stack = starts;
        while let Some(node) = stack.pop() {
            if held.contains_key(&node.key()) {
                continue;
            }
            for edge in 0..node.edges {
                if let Some((after, input)) = node.next(edge)? {
                    if let Some(feed) = edges.get(&(after.key(), input)) {
                        upstream.insert(*feed);
                    }
                    stack.push(after);
                }
            }
            held.insert(node.key(), node);
        }
        let mut seats = Vec::new();
        let mut inside: HashMap<usize, (Node<'py>, Vec<Option<(usize, u32)>>)> = HashMap::new();
        let mut stack = Vec::new();
        for root in &roots {
            let Some((node, input)) = Node::of_tensor(root)? else {
                return refused("a result that needs a gradient has no node");
            };
            seats.push((node.key(), input));
            stack.push(node);
        }
        while let Some(node) = stack.pop() {
            if inside.contains_key(&node.key()) {
                continue;
            }
            if node.flags & TCFG_NODE_PYTHON != 0 {
                return refused(format!("{} is a node written in Python", node.name));
            }
            if node.flags & TCFG_NODE_LEAF != 0 {
                return refused("a leaf that was made in the graph needs a gradient");
            }
            let mut leads = Vec::new();
            for edge in 0..node.edges {
                match node.next(edge)? {
                    Some((after, input)) => {
                        leads.push(Some((after.key(), input)));
                        if !edges.contains_key(&(after.key(), input)) {
                            stack.push(after);
                        } else {
                            held.entry(after.key()).or_insert(after);
                        }
                    }
                    None => leads.push(None),
                }
            }
            inside.insert(node.key(), (node, leads));
        }
        let mut both: Vec<(Node<'py>, Vec<Option<(usize, u32)>>)> = inside.into_values().collect();
        both.sort_by(|a, b| b.0.sequence.cmp(&a.0.sequence));
        let (order, next): (Vec<_>, Vec<_>) = both.into_iter().unzip();
        let mut slots = HashMap::new();
        let mut targets = Vec::new();
        for (at, leads) in next.iter().enumerate() {
            for (edge, lead) in leads.iter().enumerate() {
                if let Some(feed) = lead.and_then(|lead| edges.get(&lead)) {
                    slots.insert((at, edge as u32), targets.len() as u32);
                    targets.push(*feed);
                }
            }
        }
        Ok(Nodes { roots, seats, edges, upstream, order, next, slots, targets, _held: held.into_values().collect() })
    }

    /// Calls eager's nodes for the roots that come with a gradient, latest
    /// first, as the autograd engine does: the gradients that meet in front
    /// of a node are added in the order they come, and what a node gives is
    /// brought to the sizes, dtype and device of where it goes (in
    /// `node_call`). With a joint graph every kernel this runs is taken
    /// down as an operation of the backward; with `twice` the kernels run
    /// so that they can be differentiated. Returns the contributions to
    /// the gradients of the inputs, as (which, gradient), in the order they
    /// are made.
    fn driven(&self, py: Python<'py>, mut joint: Option<&mut Joint>, grads: &[Option<Bound<'py, PyAny>>], twice: bool) -> Result<Vec<(u32, Bound<'py, PyAny>)>, Stop> {
        let _mode = GradMode::set(py, twice)?;
        let mut buffers: HashMap<(usize, u32), Bound<'py, PyAny>> = HashMap::new();
        let mut reached: HashSet<usize> = HashSet::new();
        for (seat, grad) in self.seats.iter().zip(grads) {
            if let Some(grad) = grad {
                buffers.insert(*seat, grad.clone());
                reached.insert(seat.0);
            }
        }
        let none = PyDict::new(py);
        let taken_down = |joint: &mut Joint, records: Vec<Record<'py>>| -> Result<(), Stop> {
            for record in records {
                joint_says(py, joint.backward(&record.op, &record.name, &record.args, &none, &record.out))?;
            }
            Ok(())
        };
        let mut gives = Vec::new();
        for (number, node) in self.order.iter().enumerate() {
            if !reached.contains(&node.key()) {
                continue;
            }
            let handed: Vec<Option<Bound<'py, PyAny>>> = (0..node.inputs).map(|input| buffers.remove(&(node.key(), input))).collect();
            let outs = match joint.as_deref_mut() {
                Some(joint) => {
                    joint_says(py, joint.node(number as u32, handed.iter().flatten().cloned().collect()))?;
                    let tape = Tape::begin()?;
                    // What a node raises when it is called behind the
                    // forward is for the backward to raise, if one comes.
                    let outs = match node.call(&handed, twice) {
                        Ok(outs) => outs,
                        Err(error) => return refused(format!("{} raised when it was called behind the forward: {}", node.name, unsafe { crate::graph::message(error.value(py).as_ptr()) })),
                    };
                    taken_down(joint, tape.end(py)?)?;
                    outs
                }
                None => node.call(&handed, twice)?,
            };
            for (edge, lead) in self.next[number].iter().enumerate() {
                let Some(lead) = lead else {
                    continue;
                };
                reached.insert(lead.0);
                let Some(grad) = outs.get(edge).cloned().flatten() else {
                    continue;
                };
                if let Some(slot) = self.slots.get(&(number, edge as u32)) {
                    if let Some(joint) = joint.as_deref_mut() {
                        joint_says(py, joint.give(*slot, &grad))?;
                    }
                    gives.push((*slot, grad));
                    continue;
                }
                let sum = match buffers.remove(lead) {
                    None => grad,
                    Some(before) => match joint.as_deref_mut() {
                        Some(joint) => {
                            joint.adds(true);
                            let tape = Tape::begin()?;
                            let sum = before.call_method1("__add__", (grad,))?;
                            taken_down(joint, tape.end(py)?)?;
                            joint.adds(false);
                            sum
                        }
                        None => before.call_method1("__add__", (grad,))?,
                    },
                };
                buffers.insert(*lead, sum);
            }
        }
        Ok(gives)
    }

    /// Per input, what was given for it, added in the order it came.
    fn sums(&self, py: Python<'py>, gives: &[(u32, Bound<'py, PyAny>)]) -> PyResult<HashMap<u32, Bound<'py, PyAny>>> {
        let _mode = GradMode::set(py, false)?;
        let mut found: HashMap<u32, Bound<'py, PyAny>> = HashMap::new();
        for (slot, grad) in gives {
            let target = self.targets[*slot as usize];
            let sum = match found.remove(&target) {
                None => grad.clone(),
                Some(before) => before.call_method1("__add__", (grad,))?,
            };
            found.insert(target, sum);
        }
        Ok(found)
    }

    /// What the nodes gave, against what the autograd engine gives for the
    /// same tape and the same gradients: bit for bit, or the form is
    /// refused.
    fn proven(&self, py: Python<'py>, feeds: &[Bound<'py, PyAny>], roots: &[Bound<'py, PyAny>], seeds: &[Bound<'py, PyAny>], given: &HashMap<u32, Bound<'py, PyAny>>) -> Result<(), Stop> {
        let mut wanted: Vec<u32> = self.edges.values().copied().filter(|feed| !self.upstream.contains(feed)).collect();
        wanted.sort_unstable();
        wanted.dedup();
        if wanted.is_empty() {
            return Ok(());
        }
        let inputs = PyTuple::new(py, wanted.iter().map(|feed| &feeds[*feed as usize]))?;
        let engine = engine_grad(py, PyTuple::new(py, roots)?, inputs, PyTuple::new(py, seeds)?)?;
        for (feed, grad) in wanted.iter().zip(engine.try_iter()?) {
            let grad = grad?;
            let same = match given.get(feed) {
                None => grad.is_none(),
                Some(mine) => !grad.is_none() && same_bits(mine, &grad)?,
            };
            if !same {
                return refused("what eager's nodes give when the engine calls them is not what the autograd engine gives");
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// One run of a form on the tape, taken down.

/// Which value, or values, of the joint graph a value of the form's graph
/// is; for one that is several, what holds them (a tuple, a list, or a
/// named tuple of results).
enum Mark {
    Value(Value),
    Several(Vec<Mark>, Py<PyAny>),
}

impl Mark {
    fn of(joint: &Joint, value: &Bound<'_, PyAny>) -> Option<Mark> {
        if is_tensor(value) {
            return joint.value(value).map(Mark::Value);
        }
        if !(value.is_instance_of::<PyTuple>() || value.is_instance_of::<PyList>()) {
            return None;
        }
        let items: Option<Vec<Mark>> = value.try_iter().ok()?.map(|item| Mark::of(joint, &item.ok()?)).collect();
        Some(Mark::Several(items?, value.get_type().into_any().unbind()))
    }

    fn values(&self, into: &mut Vec<Value>) {
        match self {
            Mark::Value(value) => into.push(*value),
            Mark::Several(items, _) => items.iter().for_each(|item| item.values(into)),
        }
    }

    fn same(&self, other: &Mark) -> bool {
        match (self, other) {
            (Mark::Value(a), Mark::Value(b)) => a == b,
            (Mark::Several(a, kind), Mark::Several(b, other)) => a.len() == b.len() && kind.is(other) && a.iter().zip(b).all(|(a, b)| a.same(b)),
            _ => false,
        }
    }

    /// The value of the form's graph, of the tensors of the joint graph's
    /// values.
    fn made<'py>(&self, py: Python<'py>, tensor: &dyn Fn(Value) -> PyResult<Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Mark::Value(value) => tensor(*value),
            Mark::Several(items, kind) => {
                let items: Vec<Bound<'py, PyAny>> = items.iter().map(|item| item.made(py, tensor)).collect::<PyResult<_>>()?;
                let kind = kind.bind(py);
                if kind.is(&py.get_type::<PyTuple>()) {
                    Ok(PyTuple::new(py, items)?.into_any())
                } else if kind.is(&py.get_type::<PyList>()) {
                    Ok(PyList::new(py, items)?.into_any())
                } else {
                    kind.call1((PyTuple::new(py, items)?,))
                }
            }
        }
    }
}

/// How the results of a form are made of what its node gives. The node
/// gives the roots: the results that need a gradient and are no views, and
/// for a result that is a view the tensor it is a view of. The operations
/// between a root, or an input, and such a view run behind the node on
/// eager's tape (`outer`, operations of the forward): the view is then
/// eager's view, with eager's node.
#[derive(PartialEq)]
struct Shape {
    /// The values of the roots.
    roots: Vec<Value>,
    /// The values of the results that are views made behind the node.
    views: Vec<Value>,
    outer: Vec<usize>,
    /// Where the outer operations start: a value, and the root or input it
    /// is.
    sources: Vec<(Value, bool, u32)>,
    /// Per root: the result it is, or with the flag the result it is the
    /// base of.
    found: Vec<(usize, bool)>,
}

/// An input that a node of eager's holds for as long as it has what it
/// saved: the node saved the input itself, or a view of it that the graph
/// made, and a view holds what it is a view of.
#[derive(Clone, PartialEq)]
struct Keeps {
    /// The node, as the nodes run at a backward.
    node: u32,
    /// Where the node is among them in the order in which a graph that
    /// ends lets go of its nodes.
    ends: u32,
    feed: u32,
    view: bool,
}

struct Taken {
    joint: Joint,
    /// What eager's nodes keep of the inputs, in the order in which the
    /// nodes run and each lets go of what it saved.
    keeps: Vec<Keeps>,
    /// Which steps of the form's graph ran kernels.
    ran: Vec<bool>,
    /// Per value of the form's graph that is behind the kernels: what it is.
    marks: Vec<(u32, Mark)>,
    targets: Vec<u32>,
    shape: Shape,
    /// Where the form is cut in two, in place of a backward.
    cut: Option<usize>,
    /// For a form that is cut: whether its results were looked at
    /// together for a second backward (`Graph::checked` of its parts).
    checked: bool,
    /// The values of the results that something can hold behind the form.
    held: Vec<Value>,
    /// The autocast the forward ran under.
    cast: Cast,
}

impl Taken {
    fn same(&self, py: Python<'_>, other: &Taken) -> Option<String> {
        if let Some(why) = self.joint.differs(py, &other.joint) {
            return Some(why);
        }
        let marks = self.marks.len() == other.marks.len() && self.marks.iter().zip(&other.marks).all(|(a, b)| a.0 == b.0 && a.1.same(&b.1));
        let same = self.ran == other.ran && self.targets == other.targets && self.shape == other.shape && self.cut == other.cut && self.keeps == other.keeps;
        if !marks || !same || !self.cast.same(&other.cast) {
            return Some("two runs give the graph's tensors as other values".to_string());
        }
        None
    }
}

/// One kernel of the forward, for what looks at how the results were made.
struct Event<'py> {
    at: usize,
    view: bool,
    args: Bound<'py, PyTuple>,
    out: Bound<'py, PyAny>,
}

/// The input that a tensor of a run is a view of, where view operations
/// of the graph alone made it of that input.
fn viewed<'py>(events: &[Event<'py>], made: &HashMap<usize, usize>, feeds: &[Bound<'py, PyAny>], tensor: &Bound<'py, PyAny>) -> Option<usize> {
    let mut tensor = tensor.clone();
    for _ in 0..=events.len() {
        let event = &events[*made.get(&key(&tensor))?];
        if !event.view {
            return None;
        }
        tensor = event.args.get_item(0).ok().filter(is_tensor)?;
        if let Some(feed) = feeds.iter().position(|feed| feed.is(&tensor)) {
            return Some(feed);
        }
    }
    None
}

/// Where each node is in the order in which a graph that ends lets go of
/// its nodes, the roots ending in their order: a node lets go of what it
/// saved, then of the nodes that its edges alone still hold, the one
/// behind its last edge first (`deleteNode` of autograd). `numbers` says
/// which node of `order` a key is.
fn ends(nodes: &Nodes<'_>, numbers: &HashMap<usize, u32>) -> Vec<u32> {
    let inside = |lead: &Option<(usize, u32)>| lead.and_then(|(key, _)| numbers.get(&key).copied());
    let mut held = vec![0u32; nodes.order.len()];
    for to in nodes.next.iter().flatten().filter_map(inside) {
        held[to as usize] += 1;
    }
    let seats: Vec<u32> = nodes.seats.iter().filter_map(|(key, _)| numbers.get(key).copied()).collect();
    for seat in &seats {
        held[*seat as usize] += 1;
    }
    let mut places = vec![u32::MAX; nodes.order.len()];
    let mut place = 0;
    for seat in seats {
        held[seat as usize] -= 1;
        let mut stack = if held[seat as usize] == 0 { vec![seat] } else { Vec::new() };
        while let Some(node) = stack.pop() {
            places[node as usize] = place;
            place += 1;
            for to in nodes.next[node as usize].iter().filter_map(inside) {
                held[to as usize] -= 1;
                if held[to as usize] == 0 {
                    stack.push(to);
                }
            }
        }
    }
    places
}

/// What a node of eager's saved, by the names it keeps it under: the
/// tensors of each.
fn saved<'py>(node: &Node<'py>) -> PyResult<Vec<(String, Vec<Bound<'py, PyAny>>)>> {
    let mut all = Vec::new();
    for name in crate::asks::saved_names(&node.obj)? {
        let Ok(held) = node.obj.getattr(name.as_str()) else {
            continue;
        };
        let mut found = Vec::new();
        tensors_in(&held, &mut found);
        all.push((name, found));
    }
    Ok(all)
}

/// Why a graph through which eager goes backward a second time keeps
/// eager's tape.
fn twice((first, second): (String, Option<String>)) -> String {
    let which = match second {
        None => format!("a second backward for a result of this graph (a result of {first})"),
        Some(second) => format!("a backward for one result of this graph behind one for another (results of {first} and {second})"),
    };
    format!("{which} returns in eager mode where the first did not retain the graph (no node that saved a tensor runs in both), and the node of a graph is released as one")
}

/// Two roots (or one, twice) whose backwards share no node that saved a
/// tensor: eager goes backward for the second behind a backward for the
/// first that did not retain the graph, since a node raises only for
/// what it saved and has let go of. `saves` says which nodes saved one.
/// Returns the names of the nodes of the two; of the one, for one twice.
fn returns_twice(nodes: &Nodes<'_>, numbers: &HashMap<usize, u32>, saves: &[bool]) -> Option<(String, Option<String>)> {
    let inside = |lead: &Option<(usize, u32)>| lead.and_then(|(key, _)| numbers.get(&key).copied());
    let mut reached: Vec<Vec<bool>> = Vec::new();
    for (key, _) in &nodes.seats {
        let mut seen = vec![false; nodes.order.len()];
        let mut stack: Vec<u32> = numbers.get(key).copied().into_iter().collect();
        while let Some(node) = stack.pop() {
            if !std::mem::replace(&mut seen[node as usize], true) {
                stack.extend(nodes.next[node as usize].iter().filter_map(inside));
            }
        }
        reached.push(seen);
    }
    for first in 0..reached.len() {
        for second in first..reached.len() {
            if !(0..saves.len()).any(|node| saves[node] && reached[first][node] && reached[second][node]) {
                let name = |root: usize| numbers.get(&nodes.seats[root].0).map_or(String::new(), |node| nodes.order[*node as usize].name.clone());
                return Some((name(first), (first != second).then(|| name(second))));
            }
        }
    }
    None
}

/// Runs the graph of a form on the tape, one step after the other, and
/// takes the run down. Returns the values of the run, and the run as it was
/// taken down or why it was not.
fn take_down<'py>(py: Python<'py>, graph: &Graph, feeds: &[Bound<'py, PyAny>]) -> PyResult<(Vec<Option<Py<PyAny>>>, Result<Taken, String>)> {
    let mut joint = Joint::new(py, feeds.to_vec())?;
    let mut values = graph.table(feeds)?;
    // The nodes of the inputs are held while the graph runs: one that the
    // graph replaces could give its place to the one that replaces it, and
    // what adds up the gradient of a leaf lives only while something holds
    // it.
    let held: Vec<Option<(Node<'py>, u32)>> = feeds.iter().map(Node::of_tensor).collect::<PyResult<_>>()?;
    let before: Vec<(bool, Option<(usize, u32)>)> =
        feeds.iter().zip(&held).map(|(feed, node)| (requires_grad(feed), node.as_ref().map(|(node, input)| (node.key(), *input)))).collect();
    let mut ran = Vec::new();
    // The values that are the same at every call of the form: what is made
    // of constants only, and the device and the dtype of a tensor.
    let mut fixed: HashSet<u32> = HashSet::new();
    let mut events: Vec<Event<'py>> = Vec::new();
    let mut why: Option<String> = None;
    let none = PyDict::new(py);
    for step in &graph.steps {
        let tape = Tape::begin()?;
        step.run(py, &mut values)?;
        let records = tape.end(py)?;
        ran.push(!records.is_empty());
        if let Some(value) = step.value {
            let same = match step.kind {
                Kind::Tuple(..) | Kind::List(..) | Kind::Slice(..) | Kind::Same(..) | Kind::Neg(..) | Kind::Binary(..) => step.reads().iter().all(|read| fixed.contains(read)),
                Kind::Attr(..) => values[value as usize].as_ref().is_some_and(|held| kind_of(held.bind(py))),
                _ => false,
            };
            if same && records.is_empty() {
                fixed.insert(value);
            }
        }
        if why.is_some() || records.is_empty() {
            continue;
        }
        if step.plain() {
            why = Some("an operation that is no call ran a kernel".to_string());
            continue;
        }
        let mut given = Vec::new();
        for value in step.gives() {
            if let Some(held) = &values[value as usize] {
                tensors_in(held.bind(py), &mut given);
            }
        }
        // A tensor that an operation makes of what is no tensor
        // (torch.tensor of numbers, on the device and of the dtype of a
        // tensor of the graph) is filled where no kernel runs: the
        // operation itself, with what it was called with, is the one of
        // the forward that gives it. Two runs must have called it with the
        // same, and what it takes of the graph is the same at every call
        // of the form (`fixed`).
        let called = if step.reads().iter().all(|value| fixed.contains(value)) && given.len() == 1 { step.called(py, &values)? } else { None };
        if let Some((function, args, kwargs)) = called {
            let name = function.getattr("__name__").and_then(|name| name.extract::<String>()).unwrap_or_else(|_| "made".to_string());
            let random = records.iter().any(|record| record.random);
            if let Err(error) = joint.forward(&function, &format!("python::{name}"), &args, &kwargs, &given[0], Vec::new(), random) {
                why = Some(unsafe { crate::graph::message(error.value(py).as_ptr()) });
            }
            continue;
        }
        for record in records {
            if record.base() == "lift_fresh" {
                let name = match &step.kind {
                    Kind::Call(..) => step.called(py, &values)?.and_then(|(function, _, _)| function.getattr("__name__").or_else(|_| function.get_type().getattr("__name__")).ok()),
                    Kind::Aten(target, ..) => target.bind(py).getattr("__name__").ok(),
                    _ => None,
                };
                let name = name.and_then(|name| name.extract::<String>().ok()).unwrap_or_else(|| match step.kind {
                    Kind::GetItem(..) => "an index".to_string(),
                    Kind::SetItem(..) => "a store by an index".to_string(),
                    _ => "an operation".to_string(),
                });
                why = Some(format!("{name} makes a tensor of what is no tensor, and takes what is not the same at every call"));
                break;
            }
            let written: Vec<Bound<'py, PyAny>> = record.written.iter().filter_map(|at| record.args.get_item(*at).ok()).filter(is_tensor).collect();
            if let Err(error) = joint.forward(&record.op, &record.name, &record.args, &none, &record.out, written, record.random) {
                why = Some(unsafe { crate::graph::message(error.value(py).as_ptr()) });
                break;
            }
            events.push(Event { at: joint.forward_len() - 1, view: record.view, args: record.args, out: record.out });
        }
    }
    if let Some(why) = why {
        return Ok((values, Err(why)));
    }
    // An input that is written to in place with a gradient through the write
    // has another node behind the write: the tape is what says so.
    for (feed, (needed, node)) in feeds.iter().zip(&before) {
        let now = Node::of_tensor(feed)?.map(|(node, input)| (node.key(), input));
        if requires_grad(feed) != *needed || now != *node {
            return Ok((values, Err("the graph writes in place to an input, with a gradient through the write".to_string())));
        }
    }
    drop(held);
    match taken(py, graph, feeds, &values, joint, ran, &events) {
        Ok(taken) => Ok((values, Ok(taken))),
        Err(Stop::Refused(why)) => Ok((values, Err(why))),
        Err(Stop::Cut(_)) => Ok((values, Err("a view that a result is made of is read in the graph too".to_string()))),
        Err(Stop::Raised(error)) => Err(error),
    }
}

fn taken<'py>(py: Python<'py>, graph: &Graph, feeds: &[Bound<'py, PyAny>], values: &[Option<Py<PyAny>>], mut joint: Joint, ran: Vec<bool>, events: &[Event<'py>]) -> Result<Taken, Stop> {
    // What is behind the kernels reads these values of the graph: which
    // values of the joint graph they are.
    let (_, behind) = graph.behind(&ran);
    let mut marks = Vec::new();
    for value in behind {
        let held = values[value as usize].as_ref().map(|held| held.bind(py));
        match held.and_then(|held| Mark::of(&joint, held)) {
            Some(mark) => marks.push((value, mark)),
            None => return refused("an operation of the graph gives what is no tensor the engine took down"),
        }
    }
    let mut results = Vec::new();
    let mut reached: Vec<bool> = Vec::new();
    for (value, held) in graph.outputs.iter().zip(&graph.held) {
        if let Some(given) = &values[*value as usize] {
            tensors_in(given.bind(py), &mut results);
            reached.resize(results.len(), *held);
        }
    }
    let held: Vec<Value> = results.iter().zip(&reached).filter(|(_, reached)| **reached).filter_map(|(result, _)| joint.value(result)).collect();
    let unique: HashSet<usize> = results.iter().map(key).collect();
    if unique.len() != results.len() {
        return refused("a result is given twice");
    }
    for feed in feeds {
        if requires_grad(feed) && (feed.getattr("_backward_hooks")?.is_truthy()? || feed.getattr("retains_grad")?.is_truthy()?) {
            return refused("an input has a hook: taking the backward down would run it");
        }
    }
    let (shape, roots, readers) = match shaped(py, &joint, feeds, &results, &reached, events) {
        Ok(found) => found,
        Err(Stop::Cut(at)) if at > 0 => {
            // The parts of a form that is cut have a node each, released
            // apart. Whether eager goes backward a second time is a
            // question of the form's own results and of all of eager's
            // nodes for them: it is asked here, of the whole form, and
            // its parts are not asked again. Where the nodes cannot be
            // gone through as a whole, each part is asked by itself.
            let mut checked = graph.checked;
            if !checked {
                let graded: Vec<Bound<'py, PyAny>> = results.iter().filter(|result| requires_grad(result)).cloned().collect();
                match Nodes::new(feeds, graded) {
                    Ok(nodes) => {
                        let numbers: HashMap<usize, u32> = nodes.order.iter().enumerate().map(|(at, node)| (node.key(), at as u32)).collect();
                        let mut saves = Vec::new();
                        for node in &nodes.order {
                            saves.push(saved(node)?.iter().any(|(_, found)| !found.is_empty()));
                        }
                        if let Some(pair) = returns_twice(&nodes, &numbers, &saves) {
                            return refused(twice(pair));
                        }
                        checked = true;
                    }
                    Err(Stop::Raised(error)) => return Err(Stop::Raised(error)),
                    Err(_) => {}
                }
            }
            joint_says(py, joint.results(results.clone(), vec![false; results.len()]))?;
            joint.finish();
            let shape = Shape { roots: Vec::new(), views: Vec::new(), outer: Vec::new(), sources: Vec::new(), found: Vec::new() };
            return Ok(Taken { joint, keeps: Vec::new(), ran, marks, targets: Vec::new(), shape, cut: Some(at), checked, held, cast: Cast::now(py)? });
        }
        Err(stop) => return Err(stop),
    };
    let nodes = Nodes::new(feeds, roots)?;
    ordered(&nodes, &readers)?;
    // Where what each node gives goes: to another node of the graph, or
    // to an input as one of the contributions.
    let numbers: HashMap<usize, u32> = nodes.order.iter().enumerate().map(|(at, node)| (node.key(), at as u32)).collect();
    for (number, leads) in nodes.next.iter().enumerate() {
        let led = |(edge, lead): (usize, &Option<(usize, u32)>)| {
            let slot = nodes.slots.get(&(number, edge as u32)).copied();
            (if slot.is_some() { None } else { lead.and_then(|(key, _)| numbers.get(&key).copied()) }, slot)
        };
        joint.edges(number as u32, leads.iter().enumerate().map(led).collect());
    }
    let rest: Vec<Bound<'py, PyAny>> = results.iter().filter(|result| !nodes.roots.iter().any(|root| root.is(*result))).cloned().collect();
    let mut flags = vec![true; nodes.roots.len()];
    flags.extend(vec![false; rest.len()]);
    joint_says(py, joint.results(nodes.roots.iter().chain(&rest).cloned().collect(), flags.clone()))?;
    let cast = Cast::now(py)?;
    if !nodes.roots.is_empty() {
        // A backward runs under the autocast of the thread that asks for
        // it, which is none where it is asked for behind the region the
        // forward ran in, and that of the forward where it is asked for
        // inside it: it is taken down for both, as eager's nodes run then.
        let ways: &[u32] = if cast.off() { &[0] } else { &[0, 1] };
        let _steady = Steady::hold(py)?;
        for way in ways {
            let _cast = cast.set(py, *way == 0)?;
            joint.under(*way);
            // What is taken down for all the roots is used for some of
            // them: a node that is handed nothing must give nothing.
            for node in &nodes.order {
                match node.call(&vec![None; node.inputs as usize], false) {
                    Ok(given) if given.iter().all(Option::is_none) => {}
                    Ok(_) => return refused(format!("{} gives a gradient when it is handed none", node.name)),
                    Err(_) => return refused(format!("{} cannot be handed no gradient", node.name)),
                }
            }
            let seeds: Vec<Bound<'py, PyAny>> = nodes.roots.iter().map(|root| seed(py, root)).collect::<PyResult<_>>()?;
            let handed: Vec<Option<Bound<'py, PyAny>>> = seeds.iter().cloned().map(Some).collect();
            joint_says(py, joint.begin(flags.clone(), seeds.clone()))?;
            let given = nodes.driven(py, Some(&mut joint), &handed, false)?;
            joint_says(py, joint.end())?;
            nodes.proven(py, feeds, &nodes.roots, &seeds, &nodes.sums(py, &given)?)?;
            // The backwards for some of the roots are made of this one;
            // those that are checked here against the autograd engine too.
            joint.cut(false);
            let many = nodes.roots.len();
            let some: Vec<usize> = if many <= 3 { (1..(1 << many) - 1).collect() } else if many <= 8 { (0..many).map(|at| 1 << at).collect() } else { Vec::new() };
            for mask in some {
                let mut pattern: Vec<bool> = (0..many).map(|at| mask >> at & 1 == 1).collect();
                let chosen: Vec<usize> = (0..many).filter(|at| pattern[*at]).collect();
                pattern.extend(vec![false; rest.len()]);
                let (ops, incoming, gives, takes) = match joint.part_program(py, &pattern, *way, None) {
                    Ok(found) => found,
                    Err(error) if error.is_instance_of::<PyValueError>(py) => continue,
                    Err(error) => return Err(Stop::Raised(error)),
                };
                let mut inputs: Vec<Bound<'py, PyAny>> = Vec::new();
                for value in &takes {
                    inputs.push(joint.tensor(py, *value).ok_or_else(|| PyRuntimeError::new_err("torch.vm: a value of a run is gone"))?);
                }
                inputs.extend(chosen.iter().map(|at| seeds[*at].clone()));
                let all: Vec<Value> = takes.iter().chain(&incoming).copied().collect();
                let program = Program::new(ops.as_any(), joint.values(), all, gives.iter().map(|(_, value)| *value).collect())?;
                let mode = GradMode::set(py, false)?;
                // The kernels were those for the gradients of all the
                // roots: for some of them one may not fit (a view of what
                // is a sum there and something else here). Such a backward
                // is not the graph's: it is found out again where it is
                // asked for, and eager's nodes run for it.
                let Ok(run) = program.run(py, &inputs, TCFG_RUN_BELOW_AUTOGRAD) else {
                    continue;
                };
                drop(mode);
                let part: Vec<(u32, Bound<'py, PyAny>)> = gives.iter().map(|(slot, _)| *slot).zip(run).collect();
                let roots: Vec<Bound<'py, PyAny>> = chosen.iter().map(|at| nodes.roots[*at].clone()).collect();
                let given: Vec<Bound<'py, PyAny>> = chosen.iter().map(|at| seeds[*at].clone()).collect();
                nodes.proven(py, feeds, &roots, &given, &nodes.sums(py, &part)?)?;
            }
        }
    }
    // Eager's node of every value the backwards read, and which node of
    // the backward it is: what eager names where the value was written to
    // in place before a backward.
    for value in joint.read() {
        let Some(tensor) = joint.tensor(py, value) else {
            continue;
        };
        if let Some((node, output)) = Node::of_tensor(&tensor)? {
            if node.flags & TCFG_NODE_LEAF == 0 {
                joint.made_by(value, node.name.clone(), output, numbers.get(&node.key()).copied());
            }
        }
    }
    // What each node kept of the forward, in the order it looks at it when
    // it runs: by the names it keeps it under, which is also the order in
    // which it lets go of it. Of the inputs it holds those it kept, and
    // those it kept a view of.
    let mut made: HashMap<usize, usize> = HashMap::new();
    for (at, event) in events.iter().enumerate() {
        let mut given = Vec::new();
        tensors_in(&event.out, &mut given);
        made.extend(given.iter().map(|tensor| (key(tensor), at)));
    }
    let ends = ends(&nodes, &numbers);
    let mut keeps: Vec<Keeps> = Vec::new();
    let mut saves = vec![false; nodes.order.len()];
    for (number, node) in nodes.order.iter().enumerate() {
        let mut kept: Vec<Value> = Vec::new();
        for (_, found) in saved(node)? {
            saves[number] |= !found.is_empty();
            kept.extend(found.iter().filter_map(|tensor| joint.value(tensor)));
            for tensor in &found {
                let own = feeds.iter().position(|feed| feed.is(tensor));
                let Some(feed) = own.or_else(|| viewed(events, &made, feeds, tensor)) else {
                    continue;
                };
                let keep = Keeps { node: number as u32, ends: ends[number], feed: feed as u32, view: own.is_none() };
                if !keeps.contains(&keep) {
                    keeps.push(keep);
                }
            }
        }
        joint.kept_by(number as u32, kept);
    }
    if crate::graph::debug() {
        let kept: Vec<String> = keeps.iter().map(|keep| format!("{}{} by {} (node {}, ends {})", if keep.view { "a view of " } else { "" }, keep.feed, nodes.order[keep.node as usize].name, keep.node, keep.ends as i32)).collect();
        eprintln!("torch.vm graph: what eager's nodes keep of the inputs: [{}]", kept.join(", "));
    }
    // Behind a backward that did not retain the graph, eager goes backward
    // again where no node that runs has let go of something it saved. The
    // node of a graph is released as one, and raises at every backward
    // behind that: a graph through which eager would go keeps eager's tape.
    if !graph.checked {
        if let Some(pair) = returns_twice(&nodes, &numbers, &saves) {
            return refused(twice(pair));
        }
    }
    joint.finish();
    Ok(Taken { joint, keeps, ran, marks, targets: nodes.targets.clone(), shape, cut: None, checked: graph.checked, held, cast })
}

/// The roots, and how the results are made of them: for a result that is a
/// view, the operations between it and the root, or the input, it is a view
/// of. Also, per root, eager's nodes of such views that read it. `reached`
/// says per result whether something can hold it behind the form.
#[allow(clippy::type_complexity)]
fn shaped<'py>(
    py: Python<'py>,
    joint: &Joint,
    feeds: &[Bound<'py, PyAny>],
    results: &[Bound<'py, PyAny>],
    reached: &[bool],
    events: &[Event<'py>],
) -> Result<(Shape, Vec<Bound<'py, PyAny>>, HashMap<usize, Vec<Node<'py>>>), Stop> {
    let mut made: HashMap<usize, usize> = HashMap::new();
    for (at, event) in events.iter().enumerate() {
        let mut given = Vec::new();
        tensors_in(&event.out, &mut given);
        for tensor in given {
            made.insert(key(&tensor), at);
        }
    }
    let mut shape = Shape { roots: Vec::new(), views: Vec::new(), outer: Vec::new(), sources: Vec::new(), found: Vec::new() };
    let mut roots: Vec<Bound<'py, PyAny>> = Vec::new();
    let mut outer: HashSet<usize> = HashSet::new();
    let mut readers: HashMap<usize, Vec<Node<'py>>> = HashMap::new();
    let value = |tensor: &Bound<'py, PyAny>| joint.value(tensor).ok_or_else(|| Stop::Refused("a result is no value of the graph".to_string()));
    for (number, result) in results.iter().enumerate() {
        if !requires_grad(result) {
            continue;
        }
        let Some(base) = base_of(result)? else {
            if !roots.iter().any(|root| root.is(result)) {
                if node_of(result)?.is_none() {
                    return refused("a result is a leaf that was made in the graph");
                }
                roots.push(result.clone());
                shape.roots.push(value(result)?);
                shape.found.push((number, false));
            }
            continue;
        };
        let mut tensor = result.clone();
        let source = loop {
            let event = made.get(&key(&tensor)).map(|at| (*at, &events[*at]));
            let parent = event.and_then(|(_, event)| event.args.get_item(0).ok()).filter(is_tensor);
            let (Some((at, event)), Some(parent)) = (event, parent) else {
                return refused("a result is a view the engine cannot make again of what it is a view of");
            };
            if !event.view {
                return refused("a result is a view the engine cannot make again of what it is a view of");
            }
            let mut others = Vec::new();
            for arg in event.args.iter().skip(1) {
                tensors_in(&arg, &mut others);
            }
            if !others.is_empty() {
                return refused("a result is a view that is made with another tensor");
            }
            outer.insert(at);
            tensor = parent;
            if let Some(feed) = feeds.iter().position(|feed| feed.is(&tensor)) {
                break (false, feed as u32);
            }
            if base_of(&tensor)?.is_none() {
                if !tensor.is(&base) {
                    return refused("a result is a view the engine cannot make again of what it is a view of");
                }
                let which = match roots.iter().position(|root| root.is(&tensor)) {
                    Some(which) => which,
                    None => {
                        if node_of(&tensor)?.is_none() {
                            return refused("a result is a leaf that was made in the graph");
                        }
                        roots.push(tensor.clone());
                        shape.roots.push(value(&tensor)?);
                        shape.found.push((number, true));
                        roots.len() - 1
                    }
                };
                break (true, which as u32);
            }
        };
        let from = value(&tensor)?;
        if !shape.sources.iter().any(|(known, _, _)| *known == from) {
            shape.sources.push((from, source.0, source.1));
        }
        shape.views.push(value(result)?);
        // Eager's nodes of the views, down to the one that reads the root.
        if source.0 {
            let seat = Node::of_tensor(&tensor)?.map(|(node, input)| (node.key(), input));
            let mut node = Node::of_tensor(result)?.map(|(node, _)| node);
            let mut steps = 0;
            loop {
                let Some(at) = node else {
                    return refused("a result is a view whose nodes do not lead to what it is a view of");
                };
                let after = if at.edges == 1 { at.next(0)? } else { None };
                if after.as_ref().map(|(after, input)| (after.key(), *input)) == seat && seat.is_some() {
                    let known = readers.entry(source.1 as usize).or_default();
                    if !known.iter().any(|other| other.key() == at.key()) {
                        known.push(at);
                    }
                    break;
                }
                steps += 1;
                if steps > events.len() {
                    return refused("a result is a view whose nodes do not lead to what it is a view of");
                }
                node = after.map(|(after, _)| after);
            }
        }
    }
    shape.outer = outer.iter().map(|at| events[*at].at).collect();
    shape.outer.sort_unstable();
    // A view that a result is made of is eager's view only behind the node:
    // nothing in the graph may need a gradient through it. Nor through a
    // root that something can hold behind the form, as a result or as what
    // a result is a view of: whoever holds it can ask for the gradient at
    // it, keep it, hook it and change it there, so what the graph computes
    // of it comes to it from a node of its own, through its gradient
    // function. What read it before its last write is not there.
    let mut views: HashSet<usize> = HashSet::new();
    for at in &outer {
        let mut given = Vec::new();
        tensors_in(&events[*at].out, &mut given);
        views.extend(given.iter().map(key));
    }
    let mut born: HashMap<usize, usize> = HashMap::new();
    for ((number, _), root) in shape.found.iter().zip(&roots) {
        let of_root = |result: &Bound<'py, PyAny>| -> PyResult<bool> { Ok(result.is(root) || base_of(result)?.is_some_and(|base| base.is(root))) };
        let mut held = reached[*number];
        for (result, reached) in results.iter().zip(reached) {
            held = held || (*reached && of_root(result)?);
        }
        if let (true, Some(at)) = (held, made.get(&key(root))) {
            born.insert(key(root), *at);
        }
    }
    for (at, event) in events.iter().enumerate() {
        if outer.contains(&at) {
            continue;
        }
        let mut given = Vec::new();
        tensors_in(&event.out, &mut given);
        if !given.iter().any(requires_grad) {
            continue;
        }
        let mut taken = Vec::new();
        for arg in event.args.iter() {
            tensors_in(&arg, &mut taken);
        }
        if taken.iter().any(|tensor| views.contains(&key(tensor)) || born.get(&key(tensor)).is_some_and(|born| at > *born)) {
            return Err(Stop::Cut(event.at));
        }
    }
    let _ = py;
    Ok((shape, roots, readers))
}

/// The gradients that meet in front of the node of a root are added in the
/// order they come. Those of the views of a root come here in front of
/// those the graph itself has for the root; in eager they come where the
/// views were made, which must be behind everything in the graph that
/// reads the root.
fn ordered(nodes: &Nodes<'_>, readers: &HashMap<usize, Vec<Node<'_>>>) -> Result<(), Stop> {
    for (which, seat) in nodes.seats.iter().enumerate() {
        let Some(views) = readers.get(&which) else {
            continue;
        };
        let within = nodes.order.iter().zip(&nodes.next).filter(|(_, leads)| leads.iter().any(|lead| lead.as_ref() == Some(seat))).map(|(node, _)| node.sequence).max();
        let first = views.iter().map(|node| node.sequence).min();
        if let (Some(within), Some(first)) = (within, first) {
            if first < within {
                return refused("a result is a view of a tensor the graph reads behind it: gradients would be added in another order");
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// A form that was proven: its two graphs, and what runs them.

/// The backward for one set of roots that come with a gradient.
struct Back {
    run: Runner,
    /// Which contribution each value it gives is.
    slots: Vec<u32>,
    /// The values of the forward it takes, in the order it takes them.
    takes: Vec<Value>,
}

/// A value the forward keeps in memory the caller can get at, for eager's
/// text where it was written to in place before a backward.
struct Named {
    /// Where it is in what the forward keeps.
    at: usize,
    /// Eager's node that gave it when the form was taken down, and which
    /// of that node's results it is.
    name: Option<String>,
    output: u32,
    held: Held,
}

/// What of a call a value is.
enum Held {
    Feed(usize),
    Root(usize),
    /// A result that is no root.
    Result,
    /// A view, made in the graph, of an input or of a result.
    View,
}

struct Plan {
    graph: Arc<Graph>,
    joint: Joint,
    shape: Shape,
    targets: Vec<u32>,
    inputs: u32,
    /// The values the forward gives, and where each is in what it gives.
    gives: HashMap<Value, usize>,
    forward: Runner,
    /// The values the forward keeps for the backwards.
    saved: Vec<Value>,
    /// Of those, what is checked for writes at a backward.
    checked: HashMap<Value, Named>,
    /// The roots among them, and the inputs.
    watched: Vec<usize>,
    operands: Vec<usize>,
    /// Per way of running a backward and set of roots that come with a
    /// gradient: what eager's nodes look at, in the order the nodes run,
    /// each with whether the node that looks at it first is the one that
    /// gave it.
    looked: Mutex<HashMap<(u32, Vec<bool>, Vec<bool>), Arc<Vec<(Value, bool)>>>>,
    /// Per such backward: which of eager's nodes run in it; None where
    /// that is not known, which is taken as all of them.
    runs: Mutex<HashMap<(u32, Vec<bool>, Vec<bool>), Option<Arc<Vec<bool>>>>>,
    /// What eager's nodes keep of the inputs (`Taken::keeps`), and per
    /// input: which of these are for it, whether a node keeps the input
    /// itself, whether one keeps a view of it, and whether a gradient
    /// goes to it.
    keeps: Vec<Keeps>,
    keepers: Vec<Vec<u32>>,
    kept: Vec<(bool, bool)>,
    target: Vec<bool>,
    /// Per input: whether it is among what the forward keeps (`saved`).
    in_saved: Vec<bool>,
    /// Whether the forward can run again: it draws no random numbers and
    /// writes to no tensor from outside.
    again: bool,
    /// The operations behind the node, as kernels over the joint graph's
    /// values.
    outer: Vec<Step>,
    /// What is computed behind the kernels: the steps of the form's graph,
    /// and the values they take with what each is.
    behind: Vec<usize>,
    marks: Vec<(u32, Mark)>,
    /// Per way of running a backward, set of roots that come with a
    /// gradient and set of contributions that are asked for (none where
    /// all are): the backward, or why there is none.
    backs: Mutex<HashMap<(u32, Vec<bool>, Vec<bool>), Result<Arc<Back>, String>>>,
    export: Option<Py<PyAny>>,
    /// The autocast the forward ran under.
    cast: Cast,
}

/// Makes a program of operations, or with passes over graphs what the passes
/// make of them: `takes` are the values the operations are handed (as
/// `examples`), `gives` those they return.
fn runner<'py>(py: Python<'py>, export: &Option<Py<PyAny>>, ops: &Bound<'py, PyAny>, values: u32, takes: Vec<Value>, gives: Vec<Value>, examples: &[Bound<'py, PyAny>]) -> Result<Runner, Stop> {
    let Some(export) = export else {
        return Ok(Runner::Program(Program::new(ops, values, takes, gives)?));
    };
    let _mode = GradMode::set(py, false)?;
    let detached: Vec<Bound<'py, PyAny>> = examples.iter().map(|tensor| tensor.call_method0("detach")).collect::<PyResult<_>>()?;
    match export.bind(py).call1((ops, takes, gives.clone(), PyTuple::new(py, detached)?)) {
        Ok(made) => Ok(Runner::Made(made.unbind(), gives.len())),
        // What a pass raises is a reason like any other.
        Err(error) => refused(unsafe { crate::graph::said(error.value(py).as_ptr()) }),
    }
}

impl Plan {
    fn new<'py>(py: Python<'py>, graph: Arc<Graph>, taken: Taken, export: Option<Py<PyAny>>, feeds: &[Bound<'py, PyAny>]) -> Result<Plan, Stop> {
        let Taken { joint, keeps, ran, marks, targets, shape, cast, .. } = taken;
        let mut keepers: Vec<Vec<u32>> = vec![Vec::new(); feeds.len()];
        let mut kept = vec![(false, false); feeds.len()];
        for (at, keep) in keeps.iter().enumerate() {
            keepers[keep.feed as usize].push(at as u32);
            let (own, view) = &mut kept[keep.feed as usize];
            *own |= !keep.view;
            *view |= keep.view;
        }
        let target: Vec<bool> = (0..feeds.len() as u32).map(|feed| targets.contains(&feed)).collect();
        let described = joint.describe(py)?;
        let part = |name: &str| described.get_item(name)?.ok_or_else(|| PyRuntimeError::new_err("torch.vm: the joint graph does not say what it should"));
        let inputs: u32 = part("inputs")?.extract()?;
        let saved: Vec<Value> = part("saved")?.extract()?;
        let in_saved: Vec<bool> = (0..feeds.len() as u32).map(|feed| saved.contains(&feed)).collect();
        let checked: Vec<(Value, Option<String>, u32)> = part("checked")?.extract()?;
        let results: Vec<Value> = part("results")?.extract()?;
        let mut named: HashMap<Value, Named> = HashMap::new();
        for (value, name, output) in checked {
            let held = if value < inputs {
                Held::Feed(value as usize)
            } else if let Some(root) = shape.roots.iter().position(|root| *root == value) {
                Held::Root(root)
            } else if results.contains(&value) {
                Held::Result
            } else {
                Held::View
            };
            if let Some(at) = saved.iter().position(|kept| *kept == value) {
                named.insert(value, Named { at, name, output, held });
            }
        }
        let mut watched: Vec<usize> = named.values().filter_map(|named| if let Held::Root(root) = named.held { Some(root) } else { None }).collect();
        watched.sort_unstable();
        let mut operands: Vec<usize> = named.values().filter_map(|named| if let Held::Feed(feed) = named.held { Some(feed) } else { None }).collect();
        operands.sort_unstable();
        let again = !(part("draws")?.extract::<bool>()? || part("writes")?.extract::<bool>()?);
        let (behind, _) = graph.behind(&ran);
        // Behind its roots the forward gives what is kept and is no input,
        // and the tensors of what is computed behind the kernels.
        let mut wanted: Vec<Value> = shape.roots.clone();
        wanted.extend(saved.iter().copied().filter(|value| *value >= inputs));
        marks.iter().for_each(|(_, mark)| mark.values(&mut wanted));
        let mut gives: Vec<Value> = Vec::new();
        for value in wanted {
            if value >= inputs && !gives.contains(&value) && !shape.views.contains(&value) {
                gives.push(value);
            }
        }
        // The operations behind the node: each takes a root, an input, or
        // what another of them gave.
        let outer = Graph::kernels(&joint.described_forward(py, &shape.outer)?, Vec::new(), Vec::new(), Vec::new(), false)?.steps;
        let mut known: HashSet<Value> = shape.sources.iter().map(|(value, _, _)| *value).collect();
        for step in &outer {
            if !step.reads().iter().all(|value| known.contains(value)) {
                return refused("a result is a view of a tensor that was written to behind it");
            }
            known.extend(step.gives());
        }
        let all: Vec<usize> = (0..joint.forward_len()).collect();
        let ops = joint.described_forward(py, &all)?;
        let takes: Vec<Value> = (0..inputs).collect();
        let forward = runner(py, &export, ops.as_any(), joint.values(), takes, gives.clone(), feeds)?;
        Ok(Plan {
            graph,
            joint,
            shape,
            targets,
            inputs,
            gives: gives.into_iter().enumerate().map(|(at, value)| (value, at)).collect(),
            forward,
            saved,
            checked: named,
            watched,
            operands,
            looked: Mutex::new(HashMap::new()),
            runs: Mutex::new(HashMap::new()),
            keeps,
            keepers,
            kept,
            target,
            in_saved,
            again,
            outer,
            behind,
            marks,
            backs: Mutex::new(HashMap::new()),
            export,
            cast,
        })
    }

    /// The backward for the roots that came with a gradient, made when it
    /// is first asked for of the backward that was taken down.
    fn back<'py>(&self, py: Python<'py>, way: u32, pattern: &[bool], needed: &[bool], saved: &[Bound<'py, PyAny>], grads: &[Bound<'py, PyAny>]) -> Result<Arc<Back>, Stop> {
        let key = (way, pattern.to_vec(), needed.to_vec());
        if let Some(known) = self.backs.lock().unwrap().get(&key) {
            return known.clone().map_err(Stop::Refused);
        }
        // The time this takes says nothing about the function.
        crate::vm::unsettled();
        let made = self.make(py, way, pattern, needed, saved, grads);
        let kept = match &made {
            Ok(back) => Ok(back.clone()),
            Err(Stop::Refused(why)) => Err(why.clone()),
            Err(_) => return made,
        };
        // What was there ends behind the lock.
        let before = self.backs.lock().unwrap().insert(key, kept);
        drop(before);
        made
    }

    fn make<'py>(&self, py: Python<'py>, way: u32, pattern: &[bool], needed: &[bool], saved: &[Bound<'py, PyAny>], grads: &[Bound<'py, PyAny>]) -> Result<Arc<Back>, Stop> {
        let mut whole = pattern.to_vec();
        whole.resize(self.joint.results_len(), false);
        let asked = (!needed.is_empty()).then_some(needed);
        let (ops, incoming, gives, takes) = joint_says(py, self.joint.part_program(py, &whole, way, asked))?;
        let mut examples: Vec<Bound<'py, PyAny>> = Vec::new();
        for value in &takes {
            let Some(at) = self.saved.iter().position(|kept| kept == value) else {
                return Err(Stop::Raised(PyRuntimeError::new_err("torch.vm: a backward reads what the forward did not keep")));
            };
            examples.push(saved[at].clone());
        }
        examples.extend(grads.iter().cloned());
        let all: Vec<Value> = takes.iter().chain(&incoming).copied().collect();
        let run = runner(py, &self.export, ops.as_any(), self.joint.values(), all, gives.iter().map(|(_, value)| *value).collect(), &examples)?;
        Ok(Arc::new(Back { run, slots: gives.iter().map(|(slot, _)| *slot).collect(), takes }))
    }

    /// What eager's nodes look at when a backward for these roots and
    /// these contributions (none: all of them) runs.
    fn looked_at(&self, way: u32, pattern: &[bool], needed: &[bool]) -> Arc<Vec<(Value, bool)>> {
        let key = (way, pattern.to_vec(), needed.to_vec());
        if let Some(known) = self.looked.lock().unwrap().get(&key) {
            return known.clone();
        }
        let mut whole = pattern.to_vec();
        whole.resize(self.joint.results_len(), false);
        let found = Arc::new(self.joint.looked_at(&whole, way, (!needed.is_empty()).then_some(needed)));
        self.looked.lock().unwrap().insert(key, found.clone());
        found
    }

    /// Which of eager's nodes run when a backward for these roots and
    /// these contributions (none: all of them) runs; None where that is
    /// not known.
    fn ran(&self, way: u32, pattern: &[bool], needed: &[bool]) -> Option<Arc<Vec<bool>>> {
        let key = (way, pattern.to_vec(), needed.to_vec());
        if let Some(known) = self.runs.lock().unwrap().get(&key) {
            return known.clone();
        }
        let mut whole = pattern.to_vec();
        whole.resize(self.joint.results_len(), false);
        let found = self.joint.runs(&whole, way, (!needed.is_empty()).then_some(needed)).map(Arc::new);
        self.runs.lock().unwrap().insert(key, found.clone());
        found
    }

    /// The roots, of the results of another run on the tape.
    fn roots_of<'py>(&self, results: &[Bound<'py, PyAny>]) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let mut roots = Vec::new();
        for (at, based) in &self.shape.found {
            let result = results.get(*at).ok_or_else(|| PyRuntimeError::new_err("torch.vm: the graph did not run again on the tape as it was taken down"))?;
            roots.push(if *based { base_of(result)?.ok_or_else(|| PyRuntimeError::new_err("torch.vm: the graph did not run again on the tape as it was taken down"))? } else { result.clone() });
        }
        Ok(roots)
    }
}

/// What the node of a call has of an input of the graph. It holds the
/// input itself only where eager's nodes hold it, and for as long as they
/// do: what only the input holds ends where eager ends it.
enum Fed {
    /// What is no tensor.
    Plain(Py<PyAny>),
    /// A leaf that requires a gradient and that a gradient goes to: what
    /// adds its gradient up holds it for as long as the graph lives.
    Leaf(Py<PyAny>),
    /// An input that nodes of eager's keep (`Plan::keeps`), until those
    /// nodes have run in a backward that does not retain the graph; with
    /// the flag it is itself a view, which a kept view of it does not hold.
    Kept(Py<PyAny>, bool),
    /// Any other tensor: another tensor in the same memory, and what
    /// stands for the history of the input.
    Alias(Py<PyAny>, History),
    /// Nothing: let go of, or never needed.
    Gone,
}

/// What stands for the history of an input that the node of a graph
/// holds an alias of.
enum History {
    /// Nothing: no gradient goes to the input.
    None,
    /// A tensor in the memory of the input that has its history (`handle`):
    /// what a backward that is differentiated again goes through. Only for
    /// an input that no node of eager's keeps: a node that keeps an input
    /// names it, and its own node, where it was written to in place.
    Has(Py<PyAny>),
    /// The same tensor is another input of the graph, which the node holds
    /// itself (a leaf, or what a node of eager's keeps): that one.
    Same(usize),
    /// The nodes of the tape that the node keeps: they saved the input
    /// itself. The alias is let go of with that tape.
    Taped,
}

impl History {
    fn clone_ref(&self, py: Python<'_>) -> History {
        match self {
            History::None => History::None,
            History::Has(with) => History::Has(with.clone_ref(py)),
            History::Same(other) => History::Same(*other),
            History::Taped => History::Taped,
        }
    }
}

impl Fed {
    fn clone_ref(&self, py: Python<'_>) -> Fed {
        match self {
            Fed::Plain(held) => Fed::Plain(held.clone_ref(py)),
            Fed::Leaf(held) => Fed::Leaf(held.clone_ref(py)),
            Fed::Kept(held, view) => Fed::Kept(held.clone_ref(py), *view),
            Fed::Alias(held, with) => Fed::Alias(held.clone_ref(py), with.clone_ref(py)),
            Fed::Gone => Fed::Gone,
        }
    }
}

/// The tape that a backward with `create_graph` ran the forward on once
/// more: its roots, and per input that a gradient goes to the node it
/// goes to there and which of that node's inputs it is.
struct Taped {
    roots: Vec<Py<PyAny>>,
    leads: Vec<Option<(Py<PyAny>, u32)>>,
}

/// What one call of a proven form keeps for its backward. The node of the
/// call holds it.
struct Kept {
    plan: Arc<Plan>,
    feeds: Mutex<Vec<Fed>>,
    fed: Vec<u64>,
    /// Where the values of each input were at the call.
    memory: Vec<usize>,
    /// What the forward kept, with the version each tensor was at; let go
    /// when a backward that does not retain the graph has run.
    saved: Mutex<Vec<(Py<PyAny>, u64)>>,
    /// Behind a backward that does not retain the graph, until the release
    /// that follows it: which of eager's nodes it ran (`Plan::ran`; None
    /// inside: all of them). That release lets go of what those nodes
    /// kept; a release that no such backward went in front of is the
    /// graph's end.
    ran: Mutex<Option<Option<Arc<Vec<bool>>>>>,
    /// Once a backward with `create_graph` has run: its tape, whose nodes
    /// hold the inputs as eager's hold them.
    tape: Mutex<Option<Taped>>,
    /// The roots that are checked for writes (`Plan::watched`), each for
    /// as long as it lives: the node must not hold what holds the node.
    roots: Vec<Py<PyWeakrefReference>>,
    /// The inputs that are checked for writes (`Plan::operands`) and that
    /// another operation made: one that has no node any more at a backward
    /// was detached in place.
    grown: Vec<usize>,
}

/// Eager's text for a tensor that is needed for a backward and was written
/// to in place: `producer` is the node it names, and which of its results.
fn modified(tensor: &Bound<'_, PyAny>, producer: &(Option<String>, u32), expected: u64) -> PyResult<String> {
    let kind: String = tensor.call_method0("type")?.extract()?;
    let sizes: Vec<i64> = tensor.getattr("shape")?.extract()?;
    let sizes: Vec<String> = sizes.iter().map(|size| size.to_string()).collect();
    let mut text = format!("one of the variables needed for gradient computation has been modified by an inplace operation: [{kind} [{}]]", sizes.join(", "));
    if let (Some(name), output) = producer {
        // The node as eager names it there: without "Backward", and without a 0 behind it.
        let name = match name.rfind("Backward") {
            Some(at) if name[at + 8..].chars().all(|c| c.is_ascii_digit()) => {
                let digits = &name[at + 8..];
                format!("{}{}", &name[..at], if digits == "0" { "" } else { digits })
            }
            _ => name.clone(),
        };
        text += &format!(", which is output {output} of {name},");
    }
    text += &format!(" is at version {}; expected version {expected} instead.", version(tensor));
    Ok(text + " Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True, check_nan=False).")
}

/// What eager raises for a tensor that a node kept as one of its operands
/// and that has no node any more.
const DETACHED: &str = "Trying to use a saved tensor that has been detached in-place, i.e. with .detach_().This is not supported, please use out-of-place `.detach()` instead";

impl Kept {
    /// What the forward kept, with the version each tensor was at. The lock
    /// is not held while Python is asked something: another thread could
    /// wait for it with the GIL.
    fn held<'py>(&self, py: Python<'py>) -> Vec<(Bound<'py, PyAny>, u64)> {
        self.saved.lock().unwrap().iter().map(|(tensor, at)| (tensor.bind(py).clone(), *at)).collect()
    }

    /// An input of the call, if the node holds it.
    fn input<'py>(&self, py: Python<'py>, at: usize) -> Option<Bound<'py, PyAny>> {
        match self.feeds.lock().unwrap().get(at)? {
            Fed::Leaf(held) | Fed::Kept(held, _) => Some(held.bind(py).clone()),
            _ => None,
        }
    }

    /// The tensor a caller has of a value that is checked, if it lives.
    fn live<'py>(&self, py: Python<'py>, named: &Named) -> Option<Bound<'py, PyAny>> {
        match named.held {
            Held::Feed(at) => self.input(py, at),
            Held::Root(root) => {
                let at = self.plan.watched.iter().position(|watched| *watched == root)?;
                self.roots[at].bind(py).upgrade()
            }
            Held::Result | Held::View => None,
        }
    }

    /// The node eager names for a value that a node of the backward kept as
    /// one of its operands: the node the tensor has now.
    fn names(&self, py: Python<'_>, named: &Named) -> PyResult<Option<String>> {
        let Some(live) = self.live(py, named) else {
            // Eager makes a view whose memory was written to a node anew.
            let view = matches!(named.held, Held::View);
            return Ok(named.name.as_ref().map(|name| if view { "AsStridedBackward0".to_string() } else { name.clone() }));
        };
        Ok(match Node::of_tensor(&live)? {
            Some((node, _)) if node.flags & TCFG_NODE_LEAF != 0 => None,
            // The node of a graph stands for the nodes of its operations.
            Some((node, _)) if node.name == "GraphBackward" => named.name.clone().or(Some(node.name)),
            Some((node, _)) => Some(node.name),
            None => None,
        })
    }

    /// Raises what eager raises where a tensor a backward needs was written
    /// to in place behind the forward, for a backward that runs this way
    /// for these roots and these contributions.
    fn unwritten(&self, py: Python<'_>, way: u32, pattern: &[bool], needed: &[bool]) -> PyResult<()> {
        let saved = self.held(py);
        for (value, gave) in self.plan.looked_at(way, pattern, needed).iter() {
            let Some(named) = self.plan.checked.get(value) else {
                continue;
            };
            let Some((tensor, expected)) = saved.get(named.at) else {
                continue;
            };
            // A tensor that a node kept as an operand and that an operation
            // made has that operation's node: without one it was detached
            // in place, which eager does not go on from.
            let detached = match named.held {
                Held::Root(_) => self.live(py, named).is_some_and(|live| !requires_grad(&live)),
                Held::Feed(feed) => self.grown.contains(&feed) && self.input(py, feed).is_some_and(|live| !requires_grad(&live)),
                Held::Result | Held::View => false,
            };
            if !*gave && detached {
                return Err(PyRuntimeError::new_err(DETACHED));
            }
            if version(tensor) != *expected {
                // A node names itself for what it kept of its own result,
                // and for an operand the node that the tensor has by now,
                // always as its first result.
                let producer = if *gave { (named.name.clone(), named.output) } else { (self.names(py, named)?, 0) };
                return Err(PyRuntimeError::new_err(modified(tensor, &producer, *expected)?));
            }
        }
        Ok(())
    }

    /// `needed` says per contribution whether the autograd engine asks for
    /// it; where it asks for all, as a plain backward does, it is empty.
    fn backward<'py>(&self, py: Python<'py>, grads: &[Option<Bound<'py, PyAny>>], needed: &[bool], twice: bool) -> PyResult<Vec<Option<Bound<'py, PyAny>>>> {
        let mut out: Vec<Option<Bound<'py, PyAny>>> = vec![None; self.plan.targets.len()];
        if grads.iter().all(Option::is_none) {
            return Ok(out);
        }
        let pattern: Vec<bool> = grads.iter().map(Option::is_some).collect();
        if twice {
            self.unwritten(py, 0, &pattern, needed)?;
            return self.again(py, grads, out, true, "");
        }
        // The backward that was taken down as this thread runs one: with
        // no autocast, or with that of the forward.
        let cast = Cast::now(py)?;
        let way = if cast.off() {
            0
        } else if cast.same(&self.plan.cast) {
            1
        } else {
            self.unwritten(py, 0, &pattern, needed)?;
            return self.again(py, grads, out, false, "the backward runs under an autocast that the forward did not run under");
        };
        self.unwritten(py, way, &pattern, needed)?;
        let given: Vec<Bound<'py, PyAny>> = grads.iter().flatten().cloned().collect();
        let saved: Vec<Bound<'py, PyAny>> = self.held(py).into_iter().map(|(tensor, _)| tensor).collect();
        let back = match self.plan.back(py, way, &pattern, needed, &saved, &given) {
            Ok(back) => back,
            Err(Stop::Refused(why)) => return self.again(py, grads, out, false, &why),
            Err(Stop::Cut(_)) => return self.again(py, grads, out, false, "the graph is cut"),
            Err(Stop::Raised(error)) => return Err(error),
        };
        let mut inputs: Vec<Bound<'py, PyAny>> = Vec::new();
        for value in &back.takes {
            let at = self.plan.saved.iter().position(|kept| kept == value).expect("kept");
            inputs.push(saved[at].clone());
        }
        inputs.extend(given);
        let ran = match back.run.run(py, &inputs) {
            Ok(ran) => ran,
            // The backward for all the roots was compared with the autograd
            // engine's: what it raises is raised. One for some of them was
            // made of that one, with kernels that need not fit what comes
            // in here: eager's nodes run in its place, from now on.
            Err(error) if needed.is_empty() && pattern.iter().all(|came| *came) => return Err(error),
            Err(error) => {
                crate::vm::unsettled();
                let why = format!("its kernels do not fit: {}", unsafe { crate::graph::message(error.value(py).as_ptr()) }.lines().next().unwrap_or(""));
                let before = self.plan.backs.lock().unwrap().insert((way, pattern.clone(), needed.to_vec()), Err(why.clone()));
                drop(before);
                return self.again(py, grads, out, false, &why);
            }
        };
        for (slot, grad) in back.slots.iter().zip(ran) {
            out[*slot as usize] = Some(grad);
        }
        served(0);
        let ran = if retains(py)? { None } else { Some(self.plan.ran(way, &pattern, needed)) };
        *self.ran.lock().unwrap() = ran;
        Ok(out)
    }

    /// The tensors to run the forward with once more, and whether each is
    /// at the version it was fed at. An input the node holds is itself.
    /// An alias stands for an input that no gradient goes to. Where one
    /// goes to it, a backward that is not differentiated again gets a
    /// leaf in its place (the gradient at the leaf is the contribution);
    /// one that is goes through the history of the input, which the
    /// tensor that was made for it has, unless the tape of an earlier
    /// such backward is there to go through (`taped`: the versions alone
    /// are asked for). With them, what was done to an input since the
    /// call, if something was: a write in place, or other data.
    fn inputs<'py>(&self, py: Python<'py>, twice: bool, taped: bool) -> PyResult<(Vec<Bound<'py, PyAny>>, Option<&'static str>)> {
        let held: Vec<Fed> = self.feeds.lock().unwrap().iter().map(|fed| fed.clone_ref(py)).collect();
        let mut changed: Option<&'static str> = None;
        let mut feeds: Vec<Bound<'py, PyAny>> = Vec::new();
        // The same input twice is the same tensor twice.
        let mut leaves: HashMap<usize, Bound<'py, PyAny>> = HashMap::new();
        for ((fed, expected), place) in held.iter().zip(&self.fed).zip(&self.memory) {
            let feed = match fed {
                Fed::Plain(held) | Fed::Leaf(held) | Fed::Kept(held, _) | Fed::Alias(held, History::None) => held.bind(py).clone(),
                Fed::Alias(held, _) if taped => held.bind(py).clone(),
                Fed::Alias(_, History::Same(other)) => match held.get(*other) {
                    Some(Fed::Leaf(same) | Fed::Kept(same, _)) => same.bind(py).clone(),
                    _ => return Err(PyRuntimeError::new_err("torch.vm: a backward of a graph runs behind the release of what it kept")),
                },
                Fed::Alias(_, History::Has(with)) if twice => with.bind(py).clone(),
                Fed::Alias(_, History::Taped) if twice => return Err(PyRuntimeError::new_err("torch.vm: the tape of a backward of this graph is gone before an input that it stood for")),
                Fed::Alias(held, _) => match leaves.get(&(held.as_ptr() as usize)) {
                    Some(known) => known.clone(),
                    None => {
                        let made = leaf(held.bind(py))?;
                        leaves.insert(held.as_ptr() as usize, made.clone());
                        made
                    }
                },
                Fed::Gone => return Err(PyRuntimeError::new_err("torch.vm: a backward of a graph runs behind the release of what it kept")),
            };
            if version(&feed) != *expected {
                changed = changed.or(Some("was written to in place"));
            } else if memory(&feed) != *place {
                changed = changed.or(Some("was given other data (`.data = ...`)"));
            }
            feeds.push(feed);
        }
        Ok((feeds, changed))
    }

    /// A backward the backward graph does not serve: one that is to be
    /// differentiated again (`create_graph`), or one for a set of roots
    /// whose backward was not taken down. The forward is run once more, on
    /// the tape, from the inputs, and eager's nodes are called for it as
    /// when the form was taken down.
    fn again<'py>(&self, py: Python<'py>, grads: &[Option<Bound<'py, PyAny>>], mut out: Vec<Option<Bound<'py, PyAny>>>, twice: bool, why: &str) -> PyResult<Vec<Option<Bound<'py, PyAny>>>> {
        let asked = if twice { "was asked for with create_graph=True".to_string() } else { format!("is for results that came with a gradient as never before ({why})") };
        if !self.plan.again {
            return Err(PyRuntimeError::new_err(format!(
                "torch.vm: the backward of this graph {asked}, and the graph draws random numbers or writes to a tensor from outside, \
                 so it cannot be run again on the tape; torch.vm.passes.Graphs(backward=False) keeps eager's tape for it"
            )));
        }
        // A backward that is differentiated again goes through the tape
        // that the first such backward of this call ran, as eager's go
        // through the one tape of the call: its nodes say themselves what
        // they no longer have.
        let taped: Option<(Vec<Bound<'py, PyAny>>, Vec<Option<(Bound<'py, PyAny>, u32)>>)> = match (&*self.tape.lock().unwrap(), twice) {
            (Some(tape), true) => {
                let lead = |lead: &Option<(Py<PyAny>, u32)>| lead.as_ref().map(|(node, input)| (node.bind(py).clone(), *input));
                Some((tape.roots.iter().map(|root| root.bind(py).clone()).collect(), tape.leads.iter().map(lead).collect()))
            }
            _ => None,
        };
        let fresh = taped.is_none();
        let (feeds, changed) = self.inputs(py, twice, !fresh)?;
        if let Some(how) = changed {
            // The forward cannot run again from the values of the call.
            return Err(PyRuntimeError::new_err(format!("torch.vm: an input of this graph {how} before a backward that {asked}")));
        }
        let wrong = || PyRuntimeError::new_err("torch.vm: the graph did not run again on the tape as it was taken down");
        let made = match taped {
            Some((roots, leads)) => {
                let leads: PyResult<Vec<Option<(Node<'py>, u32)>>> = leads.into_iter().map(|lead| lead.map(|(node, input)| Ok((Node::of(node)?, input))).transpose()).collect();
                Nodes::over(leads?, roots)
            }
            None => {
                let graph = &self.plan.graph;
                let values = {
                    // As the forward ran: with gradients, under its autocast.
                    let _mode = GradMode::set(py, true)?;
                    let _cast = self.plan.cast.set(py, false)?;
                    graph.run(py, &feeds)?
                };
                let mut results = Vec::new();
                for value in &graph.outputs {
                    if let Some(held) = &values[*value as usize] {
                        tensors_in(held.bind(py), &mut results);
                    }
                }
                Nodes::new(&feeds, self.plan.roots_of(&results)?)
            }
        };
        let nodes = match made {
            Ok(nodes) => nodes,
            Err(Stop::Raised(error)) => return Err(error),
            Err(_) => return Err(wrong()),
        };
        if nodes.targets != self.plan.targets || nodes.roots.len() != grads.len() {
            return Err(wrong());
        }
        match nodes.driven(py, None, grads, twice) {
            Ok(gives) => {
                for (slot, grad) in gives {
                    out[slot as usize] = Some(grad);
                }
                served(if twice { 2 } else { 1 });
                let ran = if retains(py)? { None } else { Some(None) };
                *self.ran.lock().unwrap() = ran;
                if twice && fresh {
                    self.taped(py, &nodes, &feeds)?;
                }
                Ok(out)
            }
            Err(Stop::Raised(error)) => Err(error),
            Err(_) => Err(wrong()),
        }
    }

    /// Keeps the tape that a backward with `create_graph` ran: the nodes
    /// of that tape are eager's, hold the inputs that eager's nodes keep
    /// and let go of them where a backward runs through them without
    /// retaining the graph. So the node holds the tape by its roots, and
    /// of each such input an alias in the place of the input, unless the
    /// storage is watched.
    fn taped<'py>(&self, py: Python<'py>, nodes: &Nodes<'py>, feeds: &[Bound<'py, PyAny>]) -> PyResult<()> {
        let mut leads = Vec::new();
        for (at, feed) in feeds.iter().enumerate() {
            let goes = self.plan.target.get(at).copied().unwrap_or(false);
            leads.push(if goes { Node::of_tensor(feed)?.map(|(node, input)| (node.obj.unbind(), input)) } else { None });
        }
        let roots = nodes.roots.iter().map(|root| root.clone().unbind()).collect();
        let kept: Vec<(usize, Py<PyAny>)> = {
            let held = self.feeds.lock().unwrap();
            held.iter().enumerate().filter_map(|(at, fed)| if let Fed::Kept(held, _) = fed { Some((at, held.clone_ref(py))) } else { None }).collect()
        };
        let mut swaps: Vec<(usize, Py<PyAny>, Fed)> = Vec::new();
        for (at, input) in kept {
            let bound = input.bind(py);
            if !unsafe { crate::graph::storage_ends_quietly(bound.as_ptr()) } {
                continue;
            }
            let goes = requires_grad(bound) && self.plan.target.get(at).copied().unwrap_or(false);
            // The nodes of the tape saved this input: they have its history.
            let with = if goes { History::Taped } else { History::None };
            let other = alias(bound)?.unbind();
            served(4);
            swaps.push((at, input, Fed::Alias(other, with)));
        }
        // Of what the forward kept: such an input itself, and a view of one
        // (what a program makes of an input by a view operation is a view
        // of it, and holds it), each for an alias of it.
        let mut others: Vec<(Py<PyAny>, Py<PyAny>)> = Vec::new();
        for (tensor, _) in self.held(py) {
            if let Some((_, _, Fed::Alias(other, _))) = swaps.iter().find(|(_, input, _)| input.bind(py).is(&tensor)) {
                others.push((tensor.unbind(), other.clone_ref(py)));
            } else if base_of(&tensor)?.is_some_and(|base| swaps.iter().any(|(_, input, _)| input.bind(py).is(&base))) {
                let other = alias(&tensor)?.unbind();
                others.push((tensor.unbind(), other));
            }
        }
        // What was there ends behind the locks, and the tape holds it.
        let mut before: Vec<Py<PyAny>> = Vec::new();
        {
            let mut saved = self.saved.lock().unwrap();
            for (tensor, _) in saved.iter_mut() {
                if let Some((_, other)) = others.iter().find(|(known, _)| known.is(&*tensor)) {
                    before.push(std::mem::replace(tensor, other.clone_ref(py)));
                }
            }
        }
        drop(others);
        let mut gone: Vec<Fed> = Vec::new();
        {
            let mut held = self.feeds.lock().unwrap();
            for (at, _, fed) in swaps.iter() {
                gone.push(std::mem::replace(&mut held[*at], fed.clone_ref(py)));
            }
        }
        let earlier = self.tape.lock().unwrap().replace(Taped { roots, leads });
        drop(earlier);
        drop(before);
        drop(gone);
        drop(swaps);
        Ok(())
    }

    /// The autograd engine releases the node. Behind a backward that did
    /// not retain the graph: what the nodes that ran in it kept is let go
    /// of, as eager's nodes let go of what they saved, the inputs in the
    /// order in which those nodes ran and each let go; no backward of this
    /// call follows. With no such backward in front of it the graph ends:
    /// all of it is let go of, the inputs in the order in which a graph
    /// that ends lets go of its nodes.
    fn released(&self) {
        let saved = std::mem::take(&mut *self.saved.lock().unwrap());
        let tape = self.tape.lock().unwrap().take();
        let ran = self.ran.lock().unwrap().take();
        let plan = &self.plan;
        let mut goes: Vec<((u32, u32), Fed)> = Vec::new();
        let mut rest: Vec<Fed> = Vec::new();
        {
            let mut held = self.feeds.lock().unwrap();
            for (at, fed) in held.iter_mut().enumerate() {
                match fed {
                    Fed::Kept(_, view) => {
                        let view = *view;
                        let keepers = plan.keepers[at].iter().copied().filter(|keep| !plan.keeps[*keep as usize].view || !view);
                        let last = match &ran {
                            // The last of the nodes that keep it, if all of them ran.
                            Some(ran) => {
                                let run = |keep: u32| ran.as_ref().map_or(true, |ran| ran.get(plan.keeps[keep as usize].node as usize).copied().unwrap_or(true));
                                keepers.clone().all(run).then(|| (keepers.max().unwrap_or(0), 0))
                            }
                            None => Some(keepers.map(|keep| (plan.keeps[keep as usize].ends, keep)).max().unwrap_or((0, 0))),
                        };
                        if let Some(last) = last {
                            goes.push((last, std::mem::replace(fed, Fed::Gone)));
                        }
                    }
                    Fed::Alias(..) => rest.push(std::mem::replace(fed, Fed::Gone)),
                    Fed::Plain(_) | Fed::Leaf(_) | Fed::Gone => {}
                }
            }
        }
        // The inputs are in what the forward kept as well: that goes first.
        drop(saved);
        drop(tape);
        drop(rest);
        goes.sort_by_key(|(last, _)| *last);
        for (_, fed) in goes {
            drop(fed);
        }
    }

    /// The node ends: what it still holds of the inputs that eager's nodes
    /// keep is let go of in the order in which a graph that ends lets go
    /// of its nodes, each of what it saved.
    fn ended(self) {
        let Kept { plan, feeds, saved, tape, .. } = self;
        drop(saved);
        drop(tape);
        let mut goes: Vec<((u32, u32), Fed)> = Vec::new();
        let mut rest: Vec<Fed> = Vec::new();
        for (at, fed) in feeds.into_inner().unwrap().into_iter().enumerate() {
            match fed {
                Fed::Kept(_, view) => {
                    let keepers = plan.keepers[at].iter().copied().filter(|keep| !plan.keeps[*keep as usize].view || !view);
                    let last = keepers.map(|keep| (plan.keeps[keep as usize].ends, keep)).max().unwrap_or((0, 0));
                    goes.push((last, fed));
                }
                other => rest.push(other),
            }
        }
        goes.sort_by_key(|(last, _)| *last);
        for (_, fed) in goes {
            drop(fed);
        }
        drop(rest);
    }
}

/// The node is handed gradients (it holds the GIL): one per root, null where
/// none came. One new reference per contribution goes into `outs`; `needed`
/// says per contribution whether the autograd engine asks for it.
unsafe extern "C" fn node_backward(state: *mut c_void, grads: *const Obj, n_grads: u32, outs: *mut Obj, n_outs: u32, needed: *const u8, flags: u32) -> i32 {
    let kept = &*(state as *const Kept);
    let asked: Vec<bool> = (0..n_outs as usize).map(|at| *needed.add(at) != 0).collect();
    // All of them, as a plain backward asks: the backward as it is.
    let asked = if asked.iter().all(|needed| *needed) { Vec::new() } else { asked };
    Python::attach(|py| {
        let handed: Vec<Option<Bound<'_, PyAny>>> = (0..n_grads as usize)
            .map(|at| {
                let grad = *grads.add(at);
                (!grad.is_null()).then(|| Bound::<PyAny>::from_borrowed_ptr(py, grad))
            })
            .collect();
        match kept.backward(py, &handed, &asked, flags & TCFG_RUN_ON_TAPE != 0) {
            Ok(given) if given.len() == n_outs as usize => {
                for (at, item) in given.into_iter().enumerate() {
                    if let Some(item) = item {
                        *outs.add(at) = item.into_ptr();
                    }
                }
                0
            }
            Ok(_) => {
                PyRuntimeError::new_err("torch.vm: the backward of a graph gives one entry per contribution").restore(py);
                -1
            }
            Err(error) => {
                error.restore(py);
                -1
            }
        }
    })
}

/// What a backward kept is let go (`dropped` 0), or the node died (1). The
/// node holds the GIL. It is not taken here on the word of this crate's own
/// count of it: a tensor dies without the GIL under a call that has it.
unsafe extern "C" fn node_release(state: *mut c_void, dropped: i32) {
    Python::attach(|_py| {
        // Let go of behind the locks: the end of a tensor can run code.
        if dropped != 0 {
            let kept = *Box::from_raw(state as *mut Kept);
            kept.ended();
        } else {
            (*(state as *const Kept)).released();
        }
    });
}

impl Plan {
    /// What the node of a call holds of each input (`Fed`). Eager's nodes
    /// hold an input in two ways: what adds up the gradient of a leaf
    /// holds the leaf, through the edge that the gradient goes along; and
    /// a node holds what it saved, itself or as what a saved view is a
    /// view of, until a backward that does not retain the graph has run
    /// it. The node of the graph holds the input in those cases. In every
    /// other it holds nothing of it, or, where it needs its values (for
    /// what the forward keeps, for running the forward again), an alias:
    /// the values without the object, so that the object and what only it
    /// holds end where eager ends them. An alias keeps the storage, which
    /// eager would have let go of with the input: where something is told
    /// when the storage ends, or its memory is not torch's own, there is
    /// no alias to be had without being seen, and None says that this call
    /// keeps eager's tape.
    fn holds<'py>(&self, py: Python<'py>, feeds: &[Bound<'py, PyAny>]) -> PyResult<Option<Vec<Fed>>> {
        if feeds.len() != self.target.len() {
            return Err(PyTypeError::new_err(format!("a graph of {} inputs is called with {}", self.target.len(), feeds.len())));
        }
        let mut held: Vec<Fed> = Vec::with_capacity(feeds.len());
        // The same input twice has one alias.
        let mut aliased: HashMap<usize, usize> = HashMap::new();
        // The aliases, each with whether a gradient goes to its input, and
        // those that are the alias of an earlier input over again.
        let mut pending: Vec<(usize, bool)> = Vec::new();
        let mut twins: Vec<(usize, usize)> = Vec::new();
        for (at, feed) in feeds.iter().enumerate() {
            let Some(found) = info(feed) else {
                held.push(Fed::Plain(feed.clone().unbind()));
                continue;
            };
            let goes = found.requires_grad != 0 && self.target[at];
            if goes && unsafe { crate::graph::is_leaf(feed.as_ptr()) } {
                held.push(Fed::Leaf(feed.clone().unbind()));
                continue;
            }
            let (own, views) = self.kept[at];
            // A kept view holds what it is a view of: the input, unless
            // the input is a view itself.
            let view = views && unsafe { crate::graph::is_view(feed.as_ptr()) };
            if own || (views && !view) {
                held.push(Fed::Kept(feed.clone().unbind(), view));
                continue;
            }
            // The values of an input are needed for what the forward keeps
            // of it and for running the forward again: a graph that cannot
            // run again needs nothing of an input that the forward does
            // not keep.
            if !self.again && !self.in_saved[at] {
                held.push(Fed::Gone);
                continue;
            }
            if let Some(same) = aliased.get(&key(feed)) {
                twins.push((held.len(), *same));
                held.push(held[*same].clone_ref(py));
                continue;
            }
            if !unsafe { crate::graph::storage_ends_quietly(feed.as_ptr()) } {
                return Ok(None);
            }
            aliased.insert(key(feed), held.len());
            pending.push((held.len(), goes));
            held.push(Fed::Alias(alias(feed)?.unbind(), History::None));
            served(4);
        }
        if !pending.is_empty() {
            // What stands for the history of each. The same tensor can be
            // another input that the node holds itself (a leaf, or what a
            // node of eager's keeps): the history is there. Else, where a
            // gradient goes to the input, a tensor is made that has it.
            let itself: HashMap<usize, usize> = held
                .iter()
                .enumerate()
                .filter_map(|(at, fed)| match fed {
                    Fed::Leaf(own) | Fed::Kept(own, _) => Some((own.as_ptr() as usize, at)),
                    _ => None,
                })
                .collect();
            for (at, goes) in pending {
                let with = match itself.get(&key(&feeds[at])) {
                    Some(other) => History::Same(*other),
                    None if goes => History::Has(handle(&feeds[at])?.unbind()),
                    None => History::None,
                };
                if let Fed::Alias(_, history) = &mut held[at] {
                    *history = with;
                }
            }
            for (at, first) in twins {
                let with = match &held[first] {
                    Fed::Alias(_, with) => with.clone_ref(py),
                    _ => History::None,
                };
                if let Fed::Alias(_, history) = &mut held[at] {
                    *history = with;
                }
            }
        }
        Ok(Some(held))
    }

    /// One call of the form: the forward below the autograd keys, the node,
    /// the views behind it, and what the form gives.
    fn call<'py>(self: &Arc<Plan>, py: Python<'py>, feeds: &[Bound<'py, PyAny>]) -> PyResult<Vec<Bound<'py, PyAny>>> {
        // What the node is to hold of the inputs is found in front of the
        // forward: a call for which it cannot hold what it needs without
        // being seen keeps eager's tape.
        let held = if self.shape.roots.is_empty() {
            Vec::new()
        } else {
            match self.holds(py, feeds)? {
                Some(held) => held,
                None => {
                    served(3);
                    return self.graph.given(py, &self.graph.run(py, feeds)?);
                }
            }
        };
        let made = self.forward.run(py, feeds)?;
        let tensor = |value: Value| -> PyResult<Bound<'py, PyAny>> {
            if value < self.inputs {
                return Ok(feeds[value as usize].clone());
            }
            self.gives.get(&value).map(|at| made[*at].clone()).ok_or_else(|| PyRuntimeError::new_err("torch.vm: the forward of a graph did not give a value that is read behind it"))
        };
        let mut roots: Vec<Bound<'py, PyAny>> = Vec::new();
        for value in &self.shape.roots {
            let root = tensor(*value)?;
            // A root is no view; what a pass made of the forward may give one.
            roots.push(if base_of(&root)?.is_some() { copy(&root)? } else { root });
        }
        if !roots.is_empty() {
            let mut saved = Vec::new();
            for value in &self.saved {
                // Of an input that the node holds an alias of, the alias
                // is what is kept.
                let kept = match held.get(*value as usize) {
                    Some(Fed::Alias(other, _)) if *value < self.inputs => other.bind(py).clone(),
                    _ => tensor(*value)?,
                };
                // What is kept of a root is kept without its node, as eager
                // keeps what a node saves of its own result: the node holds
                // what is kept.
                let kept = if roots.iter().any(|root| root.is(&kept)) { alias(&kept)? } else { kept };
                let at = version(&kept);
                saved.push((kept.unbind(), at));
            }
            let weak: Vec<Py<PyWeakrefReference>> = self.watched.iter().map(|at| PyWeakrefReference::new(&roots[*at]).map(Bound::unbind)).collect::<PyResult<_>>()?;
            let made = |feed: &usize| -> bool {
                let feed = &feeds[*feed];
                requires_grad(feed) && !unsafe { crate::graph::is_leaf(feed.as_ptr()) }
            };
            let kept = Kept {
                plan: self.clone(),
                feeds: Mutex::new(held),
                fed: feeds.iter().map(version).collect(),
                memory: feeds.iter().map(memory).collect(),
                saved: Mutex::new(saved),
                ran: Mutex::new(None),
                tape: Mutex::new(None),
                roots: weak,
                grown: self.operands.iter().copied().filter(made).collect(),
            };
            let targets = PyTuple::new(py, self.targets.iter().map(|at| &feeds[*at as usize]))?;
            let state = Box::into_raw(Box::new(kept)) as *mut c_void;
            // The state is the node's from here on, also when the call fails.
            if unsafe { (bapi()?.graph_node)(PyTuple::new(py, &roots)?.as_ptr(), targets.as_ptr(), state, node_backward, node_release) } != 0 {
                return Err(PyErr::fetch(py));
            }
        }
        // The views of a root or of an input are made here, on eager's tape.
        let mut views: Vec<Option<Py<PyAny>>> = (0..self.joint.values()).map(|_| None).collect();
        for (value, root, which) in &self.shape.sources {
            let from = if *root { &roots[*which as usize] } else { &feeds[*which as usize] };
            views[*value as usize] = Some(from.clone().unbind());
        }
        for step in &self.outer {
            step.run(py, &mut views)?;
        }
        let last = |value: Value| -> PyResult<Bound<'py, PyAny>> {
            if let Some(which) = self.shape.roots.iter().position(|root| *root == value) {
                return Ok(roots[which].clone());
            }
            if self.shape.views.contains(&value) {
                return views[value as usize].as_ref().map(|view| view.bind(py).clone()).ok_or_else(|| PyRuntimeError::new_err("torch.vm: a view of a graph was not made"));
            }
            tensor(value)
        };
        let mut values = self.graph.table(feeds)?;
        for (value, mark) in &self.marks {
            values[*value as usize] = Some(mark.made(py, &last)?.unbind());
        }
        for at in &self.behind {
            self.graph.steps[*at].run(py, &mut values)?;
        }
        self.graph.given(py, &values)
    }
}

// ---------------------------------------------------------------------------
// A form in two parts.

/// A form in two parts, each a form of its own with a backward of its own:
/// the operations in front of the first that reads a view a result is made
/// of, and those from it on. What the first part gives the second is a
/// result of the first: a view among it is eager's view, made on the tape
/// behind the node of the first part, and the second part takes it as an
/// input like any other. The operations and their order are the form's.
struct Parts {
    graph: Arc<Graph>,
    /// Per part: the form, the values of the joint graph it takes, and
    /// those it gives.
    parts: Vec<(Arc<Form>, Vec<Value>, Vec<Value>)>,
    inputs: u32,
    values: u32,
    behind: Vec<usize>,
    marks: Vec<(u32, Mark)>,
}

impl Parts {
    fn new(py: Python<'_>, graph: Arc<Graph>, taken: Taken, export: &Option<Py<PyAny>>) -> PyResult<Parts> {
        let Taken { joint, ran, marks, cut, held, checked, .. } = taken;
        let cut = cut.expect("a form that is cut");
        let (behind, _) = graph.behind(&ran);
        let mut later: Vec<Value> = Vec::new();
        marks.iter().for_each(|(_, mark)| mark.values(&mut later));
        let count = joint.forward_len();
        let inputs: u32 = joint.describe(py)?.get_item("inputs")?.map_or(Ok(0), |found| found.extract())?;
        let mut parts = Vec::new();
        for chosen in [(0..cut).collect::<Vec<usize>>(), (cut..count).collect::<Vec<usize>>()] {
            let Some(last) = chosen.last().copied() else {
                continue;
            };
            let made: HashSet<Value> = chosen.iter().flat_map(|at| joint.forward_gives(*at).iter().copied()).collect();
            let mut takes: Vec<Value> = Vec::new();
            for at in &chosen {
                for value in joint.forward_reads(*at) {
                    if !made.contains(&value) && !takes.contains(&value) {
                        takes.push(value);
                    }
                }
            }
            let after: HashSet<Value> = (last + 1..count).flat_map(|at| joint.forward_reads(at)).collect();
            let mut gives: Vec<Value> = made.iter().copied().filter(|value| later.contains(value) || after.contains(value)).collect();
            gives.sort_unstable();
            let reached = gives.iter().map(|value| held.contains(value)).collect();
            let part = Graph::kernels(&joint.described_forward(py, &chosen)?, takes.clone(), gives.clone(), reached, checked)?;
            parts.push((Form::new(py, Arc::new(part), export), takes, gives));
        }
        Ok(Parts { graph, parts, inputs, values: joint.values(), behind, marks })
    }

    fn call<'py>(&self, py: Python<'py>, feeds: &[Bound<'py, PyAny>]) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let mut tensors: Vec<Option<Bound<'py, PyAny>>> = vec![None; self.values as usize];
        for (at, feed) in feeds.iter().enumerate().take(self.inputs as usize) {
            tensors[at] = Some(feed.clone());
        }
        let missing = || PyRuntimeError::new_err("torch.vm: a part of a graph did not give a value that is read behind it");
        for (form, takes, gives) in &self.parts {
            let handed: Vec<Bound<'py, PyAny>> = takes.iter().map(|value| tensors[*value as usize].clone().ok_or_else(missing)).collect::<PyResult<_>>()?;
            let given = form.call(py, &handed)?;
            for (value, tensor) in gives.iter().zip(given) {
                tensors[*value as usize] = Some(tensor);
            }
        }
        let last = |value: Value| tensors[value as usize].clone().ok_or_else(missing);
        let mut values = self.graph.table(feeds)?;
        for (value, mark) in &self.marks {
            values[*value as usize] = Some(mark.made(py, &last)?.unbind());
        }
        for at in &self.behind {
            self.graph.steps[*at].run(py, &mut values)?;
        }
        self.graph.given(py, &values)
    }
}

// ---------------------------------------------------------------------------
// What the engine calls for a form whose inputs need a gradient.

enum State {
    /// The runs that were taken down so far.
    Learning(Vec<Taken>),
    Proven(Arc<Plan>),
    Cut(Arc<Parts>),
    Refused(String),
}

/// For the first calls the graph on eager's tape, each taken down; then,
/// once two of them came out the same, the forward program and the node.
pub struct Form {
    graph: Arc<Graph>,
    export: Option<Py<PyAny>>,
    state: Mutex<State>,
    calls: AtomicU64,
    /// The bytes the backwards read, and those the forward keeps.
    bytes: Mutex<(u64, u64)>,
}

/// Every form that was staged and is alive, for what looks at them.
static FORMS: Mutex<Vec<Weak<Form>>> = Mutex::new(Vec::new());

impl Form {
    fn new(py: Python<'_>, graph: Arc<Graph>, export: &Option<Py<PyAny>>) -> Arc<Form> {
        let export = export.as_ref().map(|export| export.clone_ref(py));
        let form = Arc::new(Form { graph, export, state: Mutex::new(State::Learning(Vec::new())), calls: AtomicU64::new(0), bytes: Mutex::new((0, 0)) });
        let mut forms = FORMS.lock().unwrap();
        forms.retain(|form| form.strong_count() > 0);
        forms.push(Arc::downgrade(&form));
        form
    }

    fn says(&self) -> String {
        match &*self.state.lock().unwrap() {
            State::Proven(_) => "the engine's backward".to_string(),
            State::Cut(_) => "in parts".to_string(),
            State::Refused(why) => format!("eager's tape: {why}"),
            State::Learning(_) => "being taken down".to_string(),
        }
    }

    fn call<'py>(&self, py: Python<'py>, feeds: &[Bound<'py, PyAny>]) -> PyResult<Vec<Bound<'py, PyAny>>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        enum Does {
            Tape,
            Learn,
            Plan(Arc<Plan>),
            Parts(Arc<Parts>),
        }
        let does = match &*self.state.lock().unwrap() {
            State::Refused(_) => Does::Tape,
            State::Proven(plan) => Does::Plan(plan.clone()),
            State::Cut(parts) => Does::Parts(parts.clone()),
            State::Learning(_) => Does::Learn,
        };
        // Asked with the state let go of.
        let does = if matches!(does, Does::Tape) || needs_tape(py)?.is_some() { Does::Tape } else { does };
        match does {
            Does::Tape => self.graph.given(py, &self.graph.run(py, feeds)?),
            Does::Plan(plan) => plan.call(py, feeds),
            Does::Parts(parts) => parts.call(py, feeds),
            Does::Learn => self.learn(py, feeds),
        }
    }

    fn learn<'py>(&self, py: Python<'py>, feeds: &[Bound<'py, PyAny>]) -> PyResult<Vec<Bound<'py, PyAny>>> {
        // A call in which a form is taken down, or made of what was taken
        // down, is no sample of a measurement against the interpreter.
        crate::vm::unsettled();
        let (values, taken) = take_down(py, &self.graph, feeds)?;
        let given = self.graph.given(py, &values)?;
        // The state is not held while Python is asked something: another
        // thread could wait for it with the GIL.
        let (mut first, second) = {
            let mut state = self.state.lock().unwrap();
            let State::Learning(runs) = &mut *state else {
                return Ok(given);
            };
            match taken {
                Err(why) => {
                    if crate::graph::debug() {
                        eprintln!("torch.vm graph: a form of {} steps keeps eager's tape: {why}", self.graph.steps.len());
                    }
                    // What was taken down before ends behind the lock.
                    let learned = std::mem::replace(&mut *state, State::Refused(why));
                    drop(state);
                    drop(learned);
                    return Ok(given);
                }
                Ok(taken) => runs.push(taken),
            }
            if runs.len() < 2 {
                return Ok(given);
            }
            let second = runs.pop().expect("two runs");
            (runs.pop().expect("two runs"), second)
        };
        let next = match first.same(py, &second) {
            Some(why) => State::Refused(why),
            None if first.cut.is_some() => match Parts::new(py, self.graph.clone(), first, &self.export) {
                Ok(parts) => State::Cut(Arc::new(parts)),
                // What came between (an interrupt, what the handler of a
                // signal raised) says nothing about the form: it leaves
                // the call, and the form goes on learning.
                Err(error) if unsafe { crate::graph::came_between(error.value(py).as_ptr()) } => return Err(error),
                Err(error) => State::Refused(unsafe { crate::graph::message(error.value(py).as_ptr()) }),
            },
            None => {
                *self.bytes.lock().unwrap() = first.joint.cut(true);
                let export = self.export.as_ref().map(|export| export.clone_ref(py));
                match Plan::new(py, self.graph.clone(), first, export, feeds) {
                    Ok(plan) => State::Proven(Arc::new(plan)),
                    Err(Stop::Refused(why)) => State::Refused(why),
                    Err(Stop::Cut(_)) => State::Refused("the graph is cut".to_string()),
                    Err(Stop::Raised(error)) if unsafe { crate::graph::came_between(error.value(py).as_ptr()) } => return Err(error),
                    Err(Stop::Raised(error)) => State::Refused(unsafe { crate::graph::message(error.value(py).as_ptr()) }),
                }
            }
        };
        if let (true, State::Refused(why)) = (crate::graph::debug(), &next) {
            eprintln!("torch.vm graph: a form of {} steps keeps eager's tape: {why}", self.graph.steps.len());
        }
        let learned = std::mem::replace(&mut *self.state.lock().unwrap(), next);
        drop(learned);
        Ok(given)
    }
}

/// What the exporter hands the VM for a form whose inputs need a gradient:
/// called with the tensors of the form, it gives the values the form gives.
#[pyclass(module = "torch._cfg_rs", name = "Staged")]
pub struct Staged {
    form: Arc<Form>,
}

#[pymethods]
impl Staged {
    /// `description` is what the engine describes a graph with for the
    /// exporter; `export`, where the list of passes has a pass over graphs,
    /// is called with operations, the values they take and give, and
    /// tensors like those they will be handed, and returns what the passes
    /// made of the operations.
    #[new]
    #[pyo3(signature = (description, export=None))]
    fn new(py: Python<'_>, description: &Bound<'_, PyDict>, export: Option<Py<PyAny>>) -> PyResult<Staged> {
        let graph = Graph::described(py, description)?;
        Ok(Staged { form: Form::new(py, Arc::new(graph), &export) })
    }

    #[pyo3(signature = (*feeds))]
    fn __call__<'py>(&self, py: Python<'py>, feeds: &Bound<'py, PyTuple>) -> PyResult<Bound<'py, PyTuple>> {
        let feeds: Vec<Bound<'py, PyAny>> = feeds.iter().collect();
        PyTuple::new(py, self.form.call(py, &feeds)?)
    }

    /// What derives the backward of the form: "the engine's backward", "in
    /// parts", "being taken down", or "eager's tape: " and why.
    #[getter]
    fn state(&self) -> String {
        self.form.says()
    }
}

/// What the VM is handed for a graph whose backward the engine derives,
/// where the list has no pass over graphs: made of the description of the
/// graph, without Python.
pub unsafe fn staged(description: Obj) -> Result<crate::pyffi::PyObj, String> {
    let py = Python::assume_attached();
    let made = || -> PyResult<Py<Staged>> {
        let described = Bound::<PyAny>::from_borrowed_ptr(py, description);
        let described = described.cast::<PyDict>().map_err(|_| PyTypeError::new_err("the description of a graph is a dict"))?;
        Py::new(py, Staged::new(py, described, None)?)
    };
    match made() {
        Ok(form) => Ok(crate::pyffi::PyObj::from_owned(form.into_ptr())),
        Err(error) => Err(format!("the engine's backward: {}", crate::graph::message(error.value(py).as_ptr()))),
    }
}

/// How the backwards of the graphs were served since this was last asked
/// with `reset`: `graph`, by the backward graph of a form; `again` and
/// `create_graph`, by the forward run once more on the tape, without and
/// with a gradient that is differentiated again; `watched`, the calls of a
/// form that kept eager's tape because the storage of an input that the
/// node would have held an alias of is watched; `aliases`, the aliases of
/// inputs that nodes were given; `histories`, those of them that came with
/// a tensor that has the history of the input.
#[pyfunction]
#[pyo3(signature = (reset=false))]
pub fn backward_served(py: Python<'_>, reset: bool) -> PyResult<Bound<'_, PyDict>> {
    let found = PyDict::new(py);
    for (name, count) in SERVED_NAMES.iter().zip(&SERVED) {
        found.set_item(*name, if reset { count.swap(0, Ordering::Relaxed) } else { count.load(Ordering::Relaxed) })?;
    }
    Ok(found)
}

/// What every staged form that is alive runs on, how often it was called
/// since this was last asked with `reset`, and the bytes its backwards read
/// and its forward keeps.
#[pyfunction]
#[pyo3(signature = (reset=false))]
pub fn backward_forms(reset: bool) -> Vec<(String, u64, u64, u64)> {
    let forms: Vec<Arc<Form>> = FORMS.lock().unwrap().iter().filter_map(Weak::upgrade).collect();
    let mut found = Vec::new();
    for form in forms {
        let (read, kept) = *form.bytes.lock().unwrap();
        found.push((form.says(), form.calls.load(Ordering::Relaxed), read, kept));
        if reset {
            form.calls.store(0, Ordering::Relaxed);
        }
    }
    found
}
