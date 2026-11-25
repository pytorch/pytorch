// Autograd for pure blocks (see cfg/optimize.rs), replacing AOTAutograd.
//
// A pure block whose ops are all aten ops gets its own autograd node, a C++
// CfgBackward (csrc/autograd_node.cpp). The first time such a block runs with
// grad needed (grad mode on and an input requiring grad), it bootstraps: its
// origin runs eagerly on detached copies of the inputs that require grad,
// under the tracer (trace.py), so autograd records the block's graph from the
// copies. The block's tensor outputs are handed out detached, with a
// CfgBackward as their history, whose edges lead to the real inputs. Its
// backward runs autograd on the copies' graph, again traced, which gives both
// the gradients and the block's joint graph (see joint.rs).
//
// The first backward also splits the joint graph into a forward and a backward
// (cfg/partition.rs) and gives the block a training version running them (an
// Op::Autograd), guarded on what the trace depended on: grad mode, which
// params require grad, the tensor params' dtypes, devices, sizes and strides,
// and the values of scalar params. Its node keeps the values the backward
// needs (and the params, for a double backward, which runs the forward again
// on them with autograd recording).
//
// The tracer runs Python, and a backward may run inside a frame the executor
// runs, so both run outside the executor (no frames run or effects recorded).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use pyo3::ffi::PyObject;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use crate::bindings::ffi as tffi;
use crate::cfg::ir::*;

use super::graph::is_tensor;
use super::run::Outcome;

static AUTOGRAD: AtomicBool = AtomicBool::new(false);

/// Give eligible pure blocks their own autograd node (see exec/autograd.rs).
#[pyfunction]
pub fn _set_cfg_autograd(py: Python<'_>, on: bool) -> PyResult<()> {
    // Loaded here rather than at the first bootstrap, where the frame it runs
    // under might have globals without __builtins__, which imports need.
    if on {
        super::joint::trace_module(py)?;
    }
    AUTOGRAD.store(on, Ordering::Relaxed);
    Ok(())
}

pub fn enabled() -> bool {
    AUTOGRAD.load(Ordering::Relaxed)
}

// Why block b may not get its own node, if it may not. It may when it's a
// pure block (it has an origin to bootstrap from) whose only version runs
// aten ops and nothing that could read or change other state (what it reads
// is in its guard, see cfg/guard.rs).
pub fn ineligible(cfg: &Cfg, b: BlockId) -> Option<String> {
    let blk = &cfg[b];
    if blk.origin.is_none() {
        return Some("not a pure block".into());
    }
    let [ver] = &blk.versions[..] else {
        return Some(format!("{} versions", blk.versions.len()));
    };
    // A call of a pinned callee ran one torch op whenever it was observed
    // (see cfg/optimize.rs).
    let pinned_call = |v: ValueId| {
        let args = &cfg[v].args;
        matches!(cfg[v].op, Op::Call { .. }) && args.len() >= 2 && cfg[args[0]].op == Op::Null && matches!(cfg[args[1]].op, Op::Pinned(_))
    };
    let allowed = |v: ValueId| {
        matches!(
            cfg[v].op,
            Op::Aten { .. } | Op::Const(_) | Op::Lit(_) | Op::Project(_) | Op::BuildTuple(_) | Op::Resume | Op::Null | Op::Pinned(_)
        ) || pinned_call(v)
    };
    let compute = &ver.body;
    if let Some(v) = compute.iter().find(|v| !allowed(**v)) {
        let op = format!("{:?}", cfg[*v].op);
        return Some(format!("runs {}", op.split(|c: char| !c.is_alphanumeric()).next().unwrap_or("?")));
    }
    if !compute.iter().any(|v| matches!(cfg[*v].op, Op::Aten { .. }) || pinned_call(*v)) {
        return Some("runs no aten op".into());
    }
    None
}

// The last joint graph a bootstrapped block's backward traced (for testing).
static LAST_JOINT: Mutex<Option<String>> = Mutex::new(None);

/// The joint graph the last bootstrapped backward traced, printed.
#[pyfunction]
pub fn _cfg_last_joint() -> Option<String> {
    LAST_JOINT.lock().unwrap().clone()
}

// What a CfgBackward holds until its variables are released.
pub struct NodeState {
    kind: Option<Kind>,
}

enum Kind {
    Bootstrap(Box<Run>),
    Trained(Trained),
}

struct Run {
    // The block's tensor params, and the tensors it ran on: copies of those
    // that require grad.
    inputs: Vec<Py<PyAny>>,
    tensors: Vec<Py<PyAny>>,
    // Which of them are copies, and so get gradients.
    copies: Vec<usize>,
    // The block's tensor outputs as the block made them, and which of them
    // were given the node (those that require grad).
    leaves: Vec<Py<PyAny>>,
    wrap: Vec<bool>,
    // The trace: the forward's records, then (once traced) the first
    // backward's.
    records: Py<PyList>,
    fwd_records: usize,
    traced: bool,
    // The forward as a graph, once a double backward needed it.
    forward: Option<super::Prepared>,
    // Where the block is, and what its training version needs, or why it
    // can't get one.
    at: Option<Where>,
    site: Result<Site, String>,
}

struct Site {
    // What the trace depends on.
    checks: Vec<(Option<u32>, Check)>,
    tensor_params: Vec<u32>,
    // How the block's outputs are made from its params and tensor outputs.
    outputs: Vec<Template>,
}

struct Trained {
    prep: Arc<TrainingPrep>,
    // The block's tensor params and the values the backward needs, with
    // their versions when saved.
    inputs: Vec<Py<PyAny>>,
    saved: Vec<Py<PyAny>>,
    versions: Vec<i64>,
}

// A Training's graphs, prepared.
pub struct TrainingPrep {
    pub training: Arc<Training>,
    forward: super::Prepared,
    backward: super::Prepared,
}

impl TrainingPrep {
    pub fn new(training: Arc<Training>) -> PyResult<Self> {
        Ok(TrainingPrep {
            forward: super::Prepared::new(training.forward.clone(), None)?,
            backward: super::Prepared::new(training.backward.clone(), None)?,
            training,
        })
    }
}

pub fn node_apply(state: &mut NodeState, grads: &[usize], out: &mut Vec<usize>) -> bool {
    let py = unsafe { Python::assume_attached() };
    match super::outside_executor(|| apply(py, state, grads)) {
        Ok(gs) => {
            out.extend(gs.into_iter().map(|g| g.into_ptr() as usize));
            true
        }
        Err(e) => {
            e.restore(py);
            false
        }
    }
}

pub fn node_release(state: &mut NodeState) {
    state.kind = None;
}

fn apply(py: Python<'_>, state: &mut NodeState, grads: &[usize]) -> PyResult<Vec<Py<PyAny>>> {
    let Some(kind) = &mut state.kind else {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Trying to backward through the graph a second time (or directly access saved tensors after they have \
             already been freed). Specify retain_graph=True if you need to backward through the graph a second time.",
        ));
    };
    let grads: Vec<Bound<'_, PyAny>> =
        grads.iter().map(|g| unsafe { Bound::from_borrowed_ptr(py, *g as *mut PyObject) }).collect();
    match kind {
        Kind::Bootstrap(run) => bootstrap_backward(py, run, &grads),
        Kind::Trained(t) => trained_backward(py, t, &grads),
    }
}

// n gradients: `got` for the tensor params at `which`, None for the others.
fn by_param(py: Python<'_>, n: usize, which: impl IntoIterator<Item = usize>, got: Vec<Bound<'_, PyAny>>) -> Vec<Py<PyAny>> {
    let mut result: Vec<Py<PyAny>> = (0..n).map(|_| py.None()).collect();
    for (i, g) in which.into_iter().zip(got) {
        result[i] = g.unbind();
    }
    result
}

fn bound<'py>(py: Python<'py>, xs: &[Py<PyAny>]) -> Vec<Bound<'py, PyAny>> {
    xs.iter().map(|x| x.bind(py).clone()).collect()
}

fn wrapped<'a, T>(xs: &'a [T], wrap: &[bool]) -> Vec<&'a T> {
    xs.iter().zip(wrap).filter(|(_, w)| **w).map(|(x, _)| x).collect()
}

// torch.autograd.grad(outs, wrt, grads), keeping the graph; traced into
// `records` unless None.
fn grad<'py>(
    py: Python<'py>,
    outs: Vec<&Bound<'py, PyAny>>,
    wrt: Vec<&Bound<'py, PyAny>>,
    grads: &[Bound<'py, PyAny>],
    records: Option<&Bound<'py, PyList>>,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    super::joint::trace_module(py)?.call_method1("backward", (outs, wrt, grads, records))?.extract()
}

fn bootstrap_backward(py: Python<'_>, run: &mut Run, grads: &[Bound<'_, PyAny>]) -> PyResult<Vec<Py<PyAny>>> {
    let n = run.tensors.len();
    if tffi::grad_mode_enabled() {
        // create_graph: the gradients must be differentiable in the real
        // inputs, so the forward runs again on them, with autograd recording.
        if run.forward.is_none() {
            let fwd = run.records.bind(py).get_slice(0, run.fwd_records);
            let (tensors, leaves) = (bound(py, &run.tensors), bound(py, &run.leaves));
            let joint = super::joint::build(&tensors, &[], &fwd, run.fwd_records, &leaves, &[])?;
            run.forward = Some(super::Prepared::new(Arc::new(joint.cfg), None)?);
        }
        let outs = run_graph(py, run.forward.as_ref().expect("built above"), &run.inputs)?;
        let inputs = bound(py, &run.inputs);
        let got = grad(py, wrapped(&outs, &run.wrap), run.copies.iter().map(|i| &inputs[*i]).collect(), grads, None)?;
        return Ok(by_param(py, n, run.copies.iter().copied(), got));
    }
    let (tensors, leaves) = (bound(py, &run.tensors), bound(py, &run.leaves));
    let records = (!run.traced).then(|| run.records.bind(py).clone());
    let got = grad(py, wrapped(&leaves, &run.wrap), run.copies.iter().map(|i| &tensors[*i]).collect(), grads, records.as_ref())?;
    if let Some(records) = records {
        run.traced = true;
        let (text, why) = match train(py, run, grads, &records, &got) {
            Ok(text) => (text, run.site.as_ref().err().cloned()),
            Err(e) => {
                let msg = e.value(py).to_string();
                (format!("unsupported: {msg}"), Some(msg))
            }
        };
        if let (Some(why), Some(at)) = (why, &run.at) {
            super::no_training(&at.compiled, at.prep, at.block, why);
        }
        *LAST_JOINT.lock().unwrap() = Some(text);
    }
    Ok(by_param(py, n, run.copies.iter().copied(), got))
}

// Builds the joint graph from the trace and, when the block may get one,
// gives it a training version. Returns the joint graph, printed.
fn train(py: Python<'_>, run: &Run, grads: &[Bound<'_, PyAny>], records: &Bound<'_, PyList>, got: &[Bound<'_, PyAny>]) -> PyResult<String> {
    let (tensors, leaves) = (bound(py, &run.tensors), bound(py, &run.leaves));
    let joint = super::joint::build(&tensors, grads, records, run.fwd_records, &leaves, got)?;
    let text = joint.cfg.render(false, false);
    let (Ok(site), Some(at)) = (&run.site, &run.at) else {
        return Ok(text);
    };
    let split = joint.partition();
    let mut requires_grad = vec![false; run.tensors.len()];
    for i in &run.copies {
        requires_grad[*i] = true;
    }
    let training = Training {
        forward: Arc::new(split.forward),
        backward: Arc::new(split.backward),
        tensor_params: site.tensor_params.clone(),
        requires_grad,
        wrap: run.wrap.clone(),
        outputs: site.outputs.clone(),
    };
    super::add_training(&at.compiled, at.prep, at.block, site.checks.clone(), training)?;
    Ok(text)
}

fn run_graph<'py>(py: Python<'py>, prep: &super::Prepared, inputs: &[Py<PyAny>]) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let ins: Vec<*mut PyObject> = inputs.iter().map(|i| i.as_ptr()).collect();
    let outs = unsafe { super::run::run_graph(prep, &ins) }.map_err(|()| PyErr::fetch(py))?;
    Ok(outs.into_iter().map(|o| unsafe { Bound::from_owned_ptr(py, o) }).collect())
}

fn version(t: &Py<PyAny>) -> i64 {
    unsafe { tffi::tensor_from_pyobject(t.as_ptr() as usize).as_ref() }.map_or(0, tffi::tensor_version)
}

fn trained_backward(py: Python<'_>, t: &mut Trained, grads: &[Bound<'_, PyAny>]) -> PyResult<Vec<Py<PyAny>>> {
    let tr = &t.prep.training;
    let n = t.inputs.len();
    let requires = || tr.requires_grad.iter().enumerate().filter(|(_, r)| **r).map(|(i, _)| i);
    if tffi::grad_mode_enabled() {
        // create_graph: as for a bootstrapped block.
        let outs = run_graph(py, &t.prep.forward, &t.inputs)?;
        let inputs = bound(py, &t.inputs);
        let got = grad(py, wrapped(&outs, &tr.wrap), requires().map(|i| &inputs[i]).collect(), grads, None)?;
        return Ok(by_param(py, n, requires(), got));
    }
    for (x, v) in t.inputs.iter().chain(&t.saved).zip(&t.versions) {
        let now = version(x);
        if now != *v {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "one of the variables needed for gradient computation has been modified by an inplace operation: \
                 it is at version {now}; expected version {v} instead."
            )));
        }
    }
    let ins: Vec<Py<PyAny>> = t.saved.iter().map(|s| s.clone_ref(py)).chain(grads.iter().map(|g| g.clone().unbind())).collect();
    let got = run_graph(py, &t.prep.backward, &ins)?;
    Ok(by_param(py, n, requires(), got))
}

// Runs an Op::Autograd: the training forward on the block's `params`
// (borrowed), under no_grad, with a node given to its outputs that require
// grad. Returns the block's outputs (owned).
pub unsafe fn forward(tp: &Arc<TrainingPrep>, params: &[*mut PyObject]) -> PyResult<Vec<*mut PyObject>> {
    let py = Python::assume_attached();
    let tr = &tp.training;
    let inputs: Vec<Py<PyAny>> =
        tr.tensor_params.iter().map(|i| Bound::from_borrowed_ptr(py, params[*i as usize]).unbind()).collect();
    let prev = tffi::set_grad_mode(false);
    let outs = run_graph(py, &tp.forward, &inputs);
    tffi::set_grad_mode(prev);
    let mut outs = outs?;
    let saved: Vec<Py<PyAny>> = outs.split_off(tr.wrap.len()).into_iter().map(Bound::unbind).collect();
    let versions = inputs.iter().chain(&saved).map(version).collect();
    let raw: Vec<usize> = wrapped(&outs, &tr.wrap).into_iter().map(|o| o.as_ptr() as usize).collect();
    let ins: Vec<usize> = inputs.iter().map(|i| i.as_ptr() as usize).collect();
    let state = Box::new(NodeState { kind: Some(Kind::Trained(Trained { prep: tp.clone(), inputs, saved, versions })) });
    let mut got = vec![];
    let ok = tffi::autograd::make_cfg_node(state, &ins, &raw, &mut got);
    let mut got = got.into_iter().map(|w| Bound::from_owned_ptr(py, w as *mut PyObject));
    if !ok {
        return Err(PyErr::fetch(py));
    }
    let leaves: Vec<Bound<'_, PyAny>> =
        outs.into_iter().zip(&tr.wrap).map(|(o, w)| if *w { got.next().expect("one per wrapped output") } else { o }).collect();
    fn make<'py>(py: Python<'py>, t: &Template, params: &[*mut PyObject], leaves: &[Bound<'py, PyAny>]) -> PyResult<Bound<'py, PyAny>> {
        Ok(match t {
            Template::Param(i) => unsafe { Bound::from_borrowed_ptr(py, params[*i as usize]) },
            Template::Out(k) => leaves[*k as usize].clone(),
            Template::Lit(l) => super::lit_object(py, l)?.into_bound(py),
            Template::Null => unreachable!("only a whole output"),
            Template::Tuple(ts) => {
                let items = ts.iter().map(|t| make(py, t, params, leaves)).collect::<PyResult<Vec<_>>>()?;
                PyTuple::new(py, items)?.into_any()
            }
        })
    }
    tr.outputs
        .iter()
        .map(|t| match t {
            // A NULL param passed on.
            Template::Param(i) if params[*i as usize].is_null() => Ok(std::ptr::null_mut()),
            Template::Null => Ok(std::ptr::null_mut()),
            t => make(py, t, params, &leaves).map(Bound::into_ptr),
        })
        .collect()
}

// Where a bootstrapped block runs: its code's Compiled (or a clone), the
// address of the Prepared (a Plan's) it runs under, and the block.
pub struct Where {
    pub compiled: Arc<super::Compiled>,
    pub prep: usize,
    pub block: BlockId,
}

// A block's run being bootstrapped (see the top of this file).
pub struct Bootstrap {
    // The block's params, and those that are tensors (borrowed while the
    // block runs).
    params: Vec<*mut PyObject>,
    inputs: Vec<*mut PyObject>,
    run: Run,
    // (copy, the param it copies).
    copied: Vec<(*mut PyObject, *mut PyObject)>,
    tracer: Py<PyAny>,
}

impl Bootstrap {
    // Starts tracing a run of a block with `params` (borrowed); returns what
    // to run its origin on (owned). `pinned`: the params the block's guard
    // pins down already, passed through as they are.
    pub unsafe fn start(params: &[*mut PyObject], pinned: &[bool], at: Where) -> PyResult<(Bootstrap, Vec<*mut PyObject>)> {
        let py = Python::assume_attached();
        let (mut inputs, mut tensors, mut copies, mut copied, mut call) = (vec![], vec![], vec![], vec![], vec![]);
        let mut checks = vec![(None, Check::GradEnabled(true)), (None, Check::NoForwardAD), (None, Check::NoSavedTensorHooks)];
        let mut untrainable: Option<String> = None;
        let mut tensor_params = vec![];
        for (i, p) in params.iter().enumerate() {
            // CPython's NULL (an empty LoadMethod self slot): passed through.
            let Some(obj) = Bound::from_borrowed_ptr_or_opt(py, *p) else {
                if !pinned[i] {
                    untrainable.get_or_insert_with(|| format!("param {i} is NULL"));
                }
                call.push(std::ptr::null_mut());
                continue;
            };
            if pinned[i] {
                call.push(obj.into_ptr());
                continue;
            }
            let mut copy = None;
            if let Some(t) = tffi::tensor_from_pyobject(*p as usize).as_ref() {
                let rg = t.requires_grad();
                let mut m = tffi::TensorMeta::default();
                tffi::tensor_meta(t, &mut m);
                if !m.has_shape {
                    untrainable.get_or_insert_with(|| format!("tensor param {i} has no fixed shape"));
                }
                checks.push((Some(i as u32), Check::RequiresGrad(rg)));
                let (sizes, strides) = (t.sizes().to_vec(), t.strides().to_vec());
                let spec = TensorSpec { dtype: m.dtype, device_type: m.device_type, device_index: m.device_index, sizes, strides };
                checks.push((Some(i as u32), Check::TensorMeta(Box::new(spec))));
                if rg {
                    let c = obj.call_method0("detach")?.call_method0("requires_grad_")?;
                    copies.push(tensors.len());
                    copied.push((c.as_ptr(), *p));
                    copy = Some(c);
                }
                inputs.push(*p);
                tensor_params.push(i as u32);
                tensors.push(copy.clone().unwrap_or_else(|| obj.clone()).unbind());
            } else {
                let tuple = obj.is_exact_instance_of::<PyTuple>();
                let ints = || -> Option<Vec<i64>> {
                    let items: Vec<Bound<'_, PyAny>> = obj.try_iter().ok()?.collect::<PyResult<_>>().ok()?;
                    items.iter().map(|x| x.is_exact_instance_of::<pyo3::types::PyInt>().then(|| x.extract().ok()).flatten()).collect()
                };
                match super::lit(&obj) {
                    Some(l) => checks.push((Some(i as u32), Check::Value(l))),
                    None if tuple || obj.is_exact_instance_of::<PyList>() => match ints() {
                        Some(ns) => checks.push((Some(i as u32), Check::Ints(tuple, ns))),
                        None => {
                            let tp = obj.get_type().name().map_or("?".into(), |n| n.to_string());
                            untrainable.get_or_insert_with(|| format!("param {i} is a {tp} of more than ints"));
                        }
                    },
                    None => {
                        let tp = obj.get_type().name().map_or("?".into(), |n| n.to_string());
                        untrainable.get_or_insert_with(|| format!("param {i} is a {tp}, which a guard can't pin down"));
                    }
                }
            }
            call.push(copy.unwrap_or(obj).into_ptr());
        }
        let records = PyList::empty(py);
        let tracer = super::joint::trace_module(py)?.getattr("Tracer")?.call1((&records,));
        let tracer = match tracer.and_then(|t| t.call_method0("__enter__").map(|_| t)) {
            Ok(t) => t,
            Err(e) => {
                call.into_iter().for_each(|o| pyo3::ffi::Py_XDECREF(o));
                return Err(e);
            }
        };
        let site = match untrainable {
            None => Ok(Site { checks, tensor_params, outputs: vec![] }),
            Some(why) => Err(why),
        };
        let run = Run {
            inputs: inputs.iter().map(|i| Bound::from_borrowed_ptr(py, *i).unbind()).collect(),
            tensors,
            copies,
            leaves: vec![],
            wrap: vec![],
            records: records.unbind(),
            fwd_records: 0,
            traced: false,
            forward: None,
            at: Some(at),
            site,
        };
        Ok((Bootstrap { params: params.to_vec(), inputs, run, copied, tracer: tracer.unbind() }, call))
    }

    // Stops tracing, and gives the block's tensor outputs that require grad
    // the node; returns the outcome of the run with those outputs.
    pub unsafe fn finish(mut self, outcome: Outcome) -> Outcome {
        let py = Python::assume_attached();
        // A raised exception is set aside while Python code runs.
        let raised = pyo3::ffi::PyErr_GetRaisedException();
        let exited = self.tracer.bind(py).call_method1("__exit__", (py.None(), py.None(), py.None()));
        let Outcome::Leave(mut outs) = outcome else {
            drop(exited);
            pyo3::ffi::PyErr_SetRaisedException(raised);
            return outcome;
        };
        pyo3::ffi::Py_XDECREF(raised);
        match exited.and_then(|_| self.wrap(py, &mut outs)) {
            Ok(()) => Outcome::Leave(outs),
            Err(e) => {
                outs.into_iter().for_each(|o| pyo3::ffi::Py_XDECREF(o));
                e.restore(py);
                Outcome::Error
            }
        }
    }

    unsafe fn wrap(&mut self, py: Python<'_>, outs: &mut [*mut PyObject]) -> PyResult<()> {
        self.run.fwd_records = self.run.records.bind(py).len();
        // What replaces each output (or a tuple output's items, one level, as
        // for `return a, b`): itself, the real param for a copy passed
        // through, or the k-th wrapped tensor. With how a training version
        // makes it (None: it can't).
        enum New {
            Same(Py<PyAny>),
            Real(*mut PyObject),
            Wrapped(usize),
        }
        let mut raw: Vec<Py<PyAny>> = vec![];
        let (params, copied, run) = (&self.params, &self.copied, &mut self.run);
        let mut classify = |obj: Bound<'_, PyAny>| -> PyResult<(New, Option<Template>)> {
            if let Some(i) = params.iter().position(|p| *p == obj.as_ptr()) {
                return Ok((New::Same(obj.unbind()), Some(Template::Param(i as u32))));
            }
            if let Some(&(_, real)) = copied.iter().find(|(c, _)| *c == obj.as_ptr()) {
                let i = params.iter().position(|p| *p == real).expect("a copy of a param");
                return Ok((New::Real(real), Some(Template::Param(i as u32))));
            }
            if is_tensor(&obj) {
                let k = run.leaves.len() as u32;
                let rg = obj.getattr("requires_grad")?.extract::<bool>()?;
                run.wrap.push(rg);
                run.leaves.push(obj.clone().unbind());
                if rg {
                    raw.push(obj.call_method0("detach")?.unbind());
                    return Ok((New::Wrapped(raw.len() - 1), Some(Template::Out(k))));
                }
                return Ok((New::Same(obj.unbind()), Some(Template::Out(k))));
            }
            let t = super::lit(&obj).map(Template::Lit);
            Ok((New::Same(obj.unbind()), t))
        };
        let mut news = vec![];
        let mut templates = Some(vec![]);
        for o in outs.iter() {
            // CPython's NULL passed through stays as it is. A training version
            // passes on a NULL param (one its guard checks is NULL, or the
            // block is untrainable anyway).
            let Some(obj) = Bound::from_borrowed_ptr_or_opt(py, *o) else {
                news.push(None);
                let null = self.params.iter().position(|p| p.is_null()).map_or(Template::Null, |i| Template::Param(i as u32));
                if let Some(ts) = &mut templates {
                    ts.push(null);
                }
                continue;
            };
            let (new, t) = match obj.cast_exact::<PyTuple>() {
                Ok(tuple) => {
                    let (ns, ts): (Vec<New>, Vec<Option<Template>>) =
                        tuple.iter().map(&mut classify).collect::<PyResult<Vec<_>>>()?.into_iter().unzip();
                    (Err(ns), ts.into_iter().collect::<Option<Vec<_>>>().map(Template::Tuple))
                }
                Err(_) => {
                    let (n, t) = classify(obj)?;
                    (Ok(n), t)
                }
            };
            news.push(Some(new));
            templates = templates.zip(t).map(|(mut ts, t)| {
                ts.push(t);
                ts
            });
        }
        match templates {
            Some(ts) => {
                if let Ok(site) = &mut self.run.site {
                    site.outputs = ts;
                }
            }
            None => {
                if self.run.site.is_ok() {
                    self.run.site = Err("an output isn't a param, a tensor, a literal or a tuple of those".into());
                }
            }
        }
        let mut wrapped_objs: Vec<Bound<'_, PyAny>> = vec![];
        if !raw.is_empty() {
            let raw_ptrs: Vec<usize> = raw.iter().map(|r| r.as_ptr() as usize).collect();
            let ins: Vec<usize> = self.inputs.iter().map(|i| *i as usize).collect();
            let run = std::mem::replace(&mut self.run, Run::empty(py));
            let state = Box::new(NodeState { kind: Some(Kind::Bootstrap(Box::new(run))) });
            let mut got = vec![];
            let ok = tffi::autograd::make_cfg_node(state, &ins, &raw_ptrs, &mut got);
            wrapped_objs = got.iter().map(|w| Bound::from_owned_ptr(py, *w as *mut PyObject)).collect();
            if !ok {
                return Err(PyErr::fetch(py));
            }
        }
        let resolve = |n: New| -> Bound<'_, PyAny> {
            match n {
                New::Same(o) => o.into_bound(py),
                New::Real(r) => Bound::from_borrowed_ptr(py, r),
                New::Wrapped(k) => wrapped_objs[k].clone(),
            }
        };
        for (o, n) in outs.iter_mut().zip(news) {
            let Some(n) = n else { continue };
            let obj = match n {
                Ok(n) => resolve(n),
                Err(items) => PyTuple::new(py, items.into_iter().map(resolve))?.into_any(),
            };
            pyo3::ffi::Py_DECREF(std::mem::replace(o, obj.into_ptr()));
        }
        Ok(())
    }
}

impl Run {
    fn empty(py: Python<'_>) -> Run {
        Run {
            inputs: vec![],
            tensors: vec![],
            copies: vec![],
            leaves: vec![],
            wrap: vec![],
            records: PyList::empty(py).unbind(),
            fwd_records: 0,
            traced: false,
            forward: None,
            at: None,
            site: Err("released".into()),
        }
    }
}
