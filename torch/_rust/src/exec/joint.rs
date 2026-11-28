// Joint graphs: a block's forward and backward as one graph of aten ops,
// built from a trace of the ops run below Autograd (trace.py), as
// AOTAutograd builds its joint graph with make_fx.
//
// The graph's params are the forward's tensor inputs, then the tangents (the
// gradients of the outputs that require grad); it leaves with the forward's
// outputs, then the inputs' gradients (None for an input that got none). Its
// body is the forward's ops, then the backward's.
//
// It's functional: an in-place op on a tensor the graph made, which nothing
// else views, becomes its out-of-place overload, and later uses of the tensor
// read the new value. Other traces aren't supported (yet): mutating an input
// or a view, random ops (a recompute would draw new numbers), ops whose
// output depends on tensor data (.item(), nonzero), ops returning anything
// but tensors, and tensors from outside the graph.

use std::collections::{HashMap, HashSet};
use std::ffi::CString;

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyList, PyModule, PyTuple};

use crate::cfg::ir::*;

use super::graph::{is_tensor, Builder};

pub struct Joint {
    pub cfg: Cfg,
    // Each tensor value's size in bytes, and the values cheap enough to
    // recompute in the backward rather than save (pointwise ops and views).
    pub bytes: HashMap<ValueId, u64>,
    pub recomputable: HashSet<ValueId>,
    pub n_inputs: usize,
    pub n_tangents: usize,
    pub n_outputs: usize,
    // How many of the body's instrs belong to the forward.
    pub fwd_len: usize,
}

impl Joint {
    pub fn partition(&self) -> crate::cfg::partition::Split {
        crate::cfg::partition::partition(
            &self.cfg,
            self.n_inputs,
            self.n_outputs,
            self.fwd_len,
            &|v| self.bytes.get(&v).copied(),
            &|v| self.recomputable.contains(&v),
        )
    }
}

// trace.py as torch._rust._trace.
pub fn trace_module(py: Python<'_>) -> PyResult<&Bound<'_, PyModule>> {
    static MODULE: PyOnceLock<Py<PyModule>> = PyOnceLock::new();
    MODULE
        .get_or_try_init(py, || {
            let code = CString::new(include_str!("trace.py")).expect("no NUL in trace.py");
            Ok(PyModule::from_code(py, &code, c"torch/_rust/src/exec/trace.py", c"torch._rust._trace")?.unbind())
        })
        .map(|m| m.bind(py))
}

fn unsupported(msg: String) -> PyErr {
    pyo3::exceptions::PyNotImplementedError::new_err(msg)
}

// The tensors in a forward output: it, or a tuple or list's.
pub fn tensors_in<'py>(out: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    if is_tensor(out) {
        return Ok(vec![out.clone()]);
    }
    if out.is_instance_of::<PyTuple>() || out.is_instance_of::<PyList>() {
        return Ok(out.try_iter()?.collect::<PyResult<Vec<_>>>()?.into_iter().filter(is_tensor).collect());
    }
    Ok(vec![])
}

pub fn build(
    inputs: &[Bound<'_, PyAny>],
    tangents: &[Bound<'_, PyAny>],
    records: &Bound<'_, PyList>,
    fwd_records: usize,
    outputs: &[Bound<'_, PyAny>],
    grads: &[Bound<'_, PyAny>],
) -> PyResult<Joint> {
    let py = records.py();
    let tag = |n: &str| -> PyResult<Bound<'_, PyAny>> { py.import("torch")?.getattr("Tag")?.getattr(n) };
    let pointwise = tag("pointwise")?;
    let mut recomputable = HashSet::new();
    let refused_tags = [
        (tag("nondeterministic_seeded")?, "is random"),
        (tag("data_dependent_output")?, "has a data-dependent output"),
        (tag("dynamic_output_shape")?, "has a data-dependent output shape"),
    ];
    let mut b = Builder::new();
    let mut params = vec![];
    for (i, t) in inputs.iter().chain(tangents).enumerate() {
        let p = b.param(i as u32);
        b.bind(t, p)?;
        params.push(p);
    }
    let param_set: HashSet<ValueId> = params.iter().copied().collect();
    // Values some other value views (or that view another).
    let mut aliased: HashSet<ValueId> = HashSet::new();
    let mut fwd_len = 0;
    for (k, rec) in records.iter().enumerate() {
        if k == fwd_records {
            fwd_len = b.body.len();
        }
        let (func, args, kwargs, out): (Bound<'_, PyAny>, Bound<'_, PyTuple>, Bound<'_, PyDict>, Bound<'_, PyAny>) =
            rec.extract()?;
        let schema = func.getattr("_schema")?;
        let base: String = schema.getattr("name")?.extract()?;
        let overload: String = schema.getattr("overload_name")?.extract()?;
        let full = if overload.is_empty() { base.clone() } else { format!("{base}.{overload}") };
        let fail = |why: &str| unsupported(format!("can't build a joint graph: {full} {why}"));
        let tags = func.getattr("tags")?;
        for (t, why) in &refused_tags {
            if tags.contains(t)? {
                return Err(fail(why));
            }
        }
        // The objects this op writes to, by schema position.
        let sargs: Vec<Bound<'_, PyAny>> = schema.getattr("arguments")?.try_iter()?.collect::<PyResult<_>>()?;
        let mut written = vec![];
        for (i, a) in sargs.iter().enumerate() {
            let info = a.getattr("alias_info")?;
            if !info.is_none() && info.getattr("is_write")?.extract::<bool>()? {
                let name: String = a.getattr("name")?.extract()?;
                let obj = if i < args.len() { Some(args.get_item(i)?) } else { kwargs.get_item(&name)? };
                written.extend(obj);
            }
        }
        let mut name = full.clone();
        if !written.is_empty() {
            if written.len() > 1 || !base.ends_with('_') || overload.contains("out") {
                return Err(fail("mutates its args"));
            }
            let v = b.tensor(&written[0])?.ok_or_else(|| fail("mutates a tensor from outside the graph"))?;
            if param_set.contains(&v) {
                return Err(fail("mutates an input"));
            }
            if aliased.contains(&v) {
                return Err(fail("mutates a view or a viewed tensor"));
            }
            let functional = base.trim_end_matches('_');
            name = if overload.is_empty() { functional.to_string() } else { format!("{functional}.{overload}") };
            if super::resolve_aten(&name).is_err() {
                return Err(fail("has no out-of-place overload"));
            }
        }
        let mut vs: Vec<ValueId> = vec![];
        for a in args.iter() {
            vs.push(b.arg(&a).map_err(|e| fail(&e.value(py).to_string()))?);
        }
        let mut kwnames = vec![];
        for (kname, kval) in kwargs.iter() {
            kwnames.push(kname.extract::<String>()?);
            vs.push(b.arg(&kval).map_err(|e| fail(&e.value(py).to_string()))?);
        }
        let rets: Vec<Bound<'_, PyAny>> = schema.getattr("returns")?.try_iter()?.collect::<PyResult<_>>()?;
        let views = written.is_empty() && rets.iter().any(|r| r.getattr("alias_info").is_ok_and(|i| !i.is_none()));
        let cheap = views || tags.contains(&pointwise)?;
        let viewed: Vec<ValueId> = if written.is_empty() {
            if views {
                // Conservatively, every tensor arg.
                args.iter().filter(|a| is_tensor(a)).map(|a| b.tensor(&a)).collect::<PyResult<Vec<_>>>()?.into_iter().flatten().collect()
            } else {
                vec![]
            }
        } else {
            vec![]
        };
        let v = b.add(Op::Aten { name, kwnames }, vs);
        let first_project = b.body.len();
        let outs: Vec<Bound<'_, PyAny>> = if is_tensor(&out) {
            b.bind(&out, v)?;
            vec![out.clone()]
        } else if out.is_instance_of::<PyTuple>() || out.is_instance_of::<PyList>() {
            let items: Vec<Bound<'_, PyAny>> = out.try_iter()?.collect::<PyResult<_>>()?;
            if !items.iter().all(|x| is_tensor(x) || x.is_none()) {
                return Err(fail("returns non-tensors"));
            }
            for (i, x) in items.iter().enumerate() {
                let p = b.add(Op::Project(i as u32), vec![v]);
                if is_tensor(x) {
                    b.bind(x, p)?;
                }
            }
            items
        } else {
            return Err(fail("returns non-tensors"));
        };
        if cheap {
            recomputable.insert(v);
            recomputable.extend(b.body[first_project..].iter().copied());
        }
        if !viewed.is_empty() {
            aliased.extend(viewed);
            for o in outs.iter().filter(|o| is_tensor(o)) {
                aliased.extend(b.tensor(o)?);
            }
        }
        if let Some(w) = written.first() {
            // In place: the tensor it wrote now holds the new value.
            b.bind(w, b.tensor(&out)?.expect("bound above"))?;
        }
    }
    if fwd_records == records.len() {
        fwd_len = b.body.len();
    }
    let mut outs = vec![];
    for o in outputs.iter().chain(grads) {
        outs.push(b.arg(o).map_err(|e| unsupported(format!("can't build a joint graph: an output {}", e.value(py))))?);
    }
    let bytes = std::mem::take(&mut b.bytes);
    Ok(Joint {
        cfg: b.finish(params, outs)?,
        bytes,
        recomputable,
        n_inputs: inputs.len(),
        n_tangents: tangents.len(),
        n_outputs: outputs.len(),
        fwd_len,
    })
}

/// Traces fn(*args) and its backward (with ones as the gradients of its tensor
/// outputs that require grad) and prints the joint graph, or with
/// partition=True the forward and backward it splits into. For testing.
#[pyfunction]
#[pyo3(signature = (fn_, *args, partition=false))]
pub fn _trace_joint(fn_: &Bound<'_, PyAny>, args: Vec<Bound<'_, PyAny>>, partition: bool) -> PyResult<String> {
    let py = fn_.py();
    let trace = trace_module(py)?;
    let records = PyList::empty(py);
    type Forward<'py> = (Vec<Option<Bound<'py, PyAny>>>, Vec<Bound<'py, PyAny>>, Bound<'py, PyAny>);
    let (inner, call, out): Forward<'_> = trace.call_method1("forward", (fn_, args, &records))?.extract()?;
    let fwd_records = records.len();
    let outputs = tensors_in(&out)?;
    let needs_grad: Vec<Bound<'_, PyAny>> =
        outputs.iter().filter(|o| o.getattr("requires_grad").and_then(|r| r.extract()).unwrap_or(false)).cloned().collect();
    let torch = py.import("torch")?;
    let tangents: Vec<Bound<'_, PyAny>> =
        needs_grad.iter().map(|o| torch.call_method1("ones_like", (o,))).collect::<PyResult<_>>()?;
    let inner: Vec<Bound<'_, PyAny>> = inner.into_iter().flatten().collect();
    let grads: Vec<Bound<'_, PyAny>> = if needs_grad.is_empty() {
        vec![]
    } else {
        trace.call_method1("backward", (&needs_grad, &inner, &tangents, &records))?.extract()?
    };
    let inputs: Vec<Bound<'_, PyAny>> = call.into_iter().filter(is_tensor).collect();
    let joint = build(&inputs, &tangents, &records, fwd_records, &outputs, &grads)?;
    if partition {
        let split = joint.partition();
        return Ok(format!("forward:\n{}backward:\n{}", split.forward.render(false, false), split.backward.render(false, false)));
    }
    Ok(format!(
        "# {} inputs, {} tangents, {} outputs; the first {} instrs are the forward\n{}",
        joint.n_inputs,
        joint.n_tangents,
        joint.n_outputs,
        joint.fwd_len,
        joint.cfg.render(false, false)
    ))
}
