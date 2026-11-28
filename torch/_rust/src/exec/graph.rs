// Graphs: straight-line Cfgs of aten ops (a traced forward or backward), built
// from an op list and run without a frame (run::run_graph).
//
// An op list is what a tracer records: `nparams` inputs, then ops of the form
// (name, args, kwargs, nouts). Values are numbered in order: the inputs, then
// each op's outputs, one for nouts None (a single return) or nouts of them for
// a returned tuple or list. An arg is a Ref to a value, a list or tuple
// holding Refs (built at run time), or a literal (see Lit).

use std::collections::HashMap;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::cfg::ir::*;

use super::Prepared;

/// A reference to a value of an op list: input `idx`, or an op's output.
#[pyclass(frozen, module = "torch._rust")]
pub struct Ref {
    #[pyo3(get)]
    idx: u32,
}

#[pymethods]
impl Ref {
    #[new]
    fn new(idx: u32) -> Self {
        Ref { idx }
    }

    fn __repr__(&self) -> String {
        format!("Ref({})", self.idx)
    }
}

pub(super) struct Builder {
    pub cfg: Cfg,
    pub body: Vec<ValueId>,
    // By Ref index.
    pub vals: Vec<ValueId>,
    literals: HashMap<Lit, ValueId>,
    // A traced tensor's value, by object, and by where its data is, for a
    // tensor that autograd re-wrapped (a saved output unpacks as a new
    // tensor object over the same data).
    tensors: HashMap<usize, ValueId>,
    storage: HashMap<StorageKey, ValueId>,
    // Each traced tensor value's size in bytes.
    pub bytes: HashMap<ValueId, u64>,
}

// (storage data pointer, offset, sizes, strides, dtype).
type StorageKey = (usize, i64, Vec<i64>, Vec<i64>, String);

fn storage_key(t: &Bound<'_, PyAny>) -> PyResult<StorageKey> {
    Ok((
        t.call_method0("untyped_storage")?.call_method0("data_ptr")?.extract()?,
        t.call_method0("storage_offset")?.extract()?,
        t.call_method0("size")?.extract()?,
        t.call_method0("stride")?.extract()?,
        t.getattr("dtype")?.str()?.extract()?,
    ))
}

pub(super) fn is_tensor(a: &Bound<'_, PyAny>) -> bool {
    !unsafe { crate::bindings::ffi::tensor_from_pyobject(a.as_ptr() as usize) }.is_null()
}

impl Builder {
    pub fn new() -> Self {
        Builder {
            cfg: Cfg::new(vec![], vec![], vec![]),
            body: vec![],
            vals: vec![],
            literals: HashMap::new(),
            tensors: HashMap::new(),
            storage: HashMap::new(),
            bytes: HashMap::new(),
        }
    }

    fn lit(&mut self, l: Lit) -> ValueId {
        if let Some(v) = self.literals.get(&l) {
            return *v;
        }
        let i = self.cfg.literals.len() as u32;
        self.cfg.literals.push(l.clone());
        let v = self.add(Op::Lit(i), vec![]);
        self.literals.insert(l, v);
        v
    }

    pub fn add(&mut self, op: Op, args: Vec<ValueId>) -> ValueId {
        let v = self.cfg.add(op, args, 0);
        self.body.push(v);
        v
    }

    // Makes `t` (a tensor) stand for v from now on.
    pub fn bind(&mut self, t: &Bound<'_, PyAny>, v: ValueId) -> PyResult<()> {
        self.tensors.insert(t.as_ptr() as usize, v);
        self.storage.insert(storage_key(t)?, v);
        let numel: u64 = t.call_method0("numel")?.extract()?;
        let size: u64 = t.call_method0("element_size")?.extract()?;
        self.bytes.insert(v, numel * size);
        Ok(())
    }

    pub fn tensor(&self, t: &Bound<'_, PyAny>) -> PyResult<Option<ValueId>> {
        if let Some(v) = self.tensors.get(&(t.as_ptr() as usize)) {
            return Ok(Some(*v));
        }
        Ok(self.storage.get(&storage_key(t)?).copied())
    }

    fn get(&self, r: &Ref) -> PyResult<ValueId> {
        self.vals.get(r.idx as usize).copied().ok_or_else(|| err(format!("Ref({}) is not defined yet", r.idx)))
    }

    pub fn arg(&mut self, a: &Bound<'_, PyAny>) -> PyResult<ValueId> {
        if let Ok(r) = a.cast::<Ref>() {
            return self.get(r.get());
        }
        if is_tensor(a) {
            return self.tensor(a)?.ok_or_else(|| err("uses a tensor from outside the graph".into()));
        }
        if a.is_instance_of::<PyList>() || a.is_instance_of::<PyTuple>() {
            let items: Vec<Bound<'_, PyAny>> = a.try_iter()?.collect::<PyResult<_>>()?;
            if items.iter().any(|x| x.is_instance_of::<Ref>() || is_tensor(x)) {
                let args = items.iter().map(|x| self.arg(x)).collect::<PyResult<_>>()?;
                return Ok(self.add(Op::BuildTuple(items.len() as u32), args));
            }
        }
        let l = lit(a)?.ok_or_else(|| err(format!("can't express {} as a literal", a.repr().map_or("?".into(), |r| r.to_string()))))?;
        Ok(self.lit(l))
    }

    // The graph: one block with `params` that leaves with `outputs`, verified.
    pub fn finish(mut self, params: Vec<ValueId>, outputs: Vec<ValueId>) -> PyResult<Cfg> {
        let mut blk = Block::new();
        blk.params = params;
        let n = outputs.len();
        blk.versions.push(Version { body: self.body, outputs, ..Version::default() });
        blk.term = Terminator::Jump(Exit::Leave((0..n as u32).collect()));
        self.cfg.boundary = Some(Boundary { n_outputs: n, handler_arity: vec![] });
        self.cfg.blocks.push(blk);
        crate::cfg::verify::verify(&self.cfg)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("graph verification failed: {e}")))?;
        Ok(self.cfg)
    }

    pub fn param(&mut self, i: u32) -> ValueId {
        self.cfg.add(Op::Param(ParamKind::Stack(i)), vec![], 0)
    }
}

pub(super) fn err(msg: String) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(msg)
}

// A Python value an aten op takes, as a Lit.
fn lit(o: &Bound<'_, PyAny>) -> PyResult<Option<Lit>> {
    use pyo3::types::{PyBool, PyFloat, PyInt, PyString};
    if let Some(l) = super::lit(o) {
        return Ok(Some(l));
    }
    if o.is_exact_instance_of::<PyString>() {
        return Ok(Some(Lit::Str(o.extract()?)));
    }
    let torch = o.py().import("torch")?;
    let named = |t: &str| -> PyResult<Option<String>> {
        if o.get_type().is(&torch.getattr(t)?) {
            // str(torch.float32) is "torch.float32".
            let s: String = o.str()?.extract()?;
            return Ok(Some(s.strip_prefix("torch.").unwrap_or(&s).to_string()));
        }
        Ok(None)
    };
    if let Some(n) = named("dtype")? {
        return Ok(Some(Lit::Dtype(n)));
    }
    if let Some(n) = named("layout")? {
        return Ok(Some(Lit::Layout(n)));
    }
    if let Some(n) = named("memory_format")? {
        return Ok(Some(Lit::MemoryFormat(n)));
    }
    if o.get_type().is(&torch.getattr("device")?) {
        return Ok(Some(Lit::Device(o.str()?.extract()?)));
    }
    if o.is_instance_of::<PyList>() || o.is_instance_of::<PyTuple>() {
        let items: Vec<Bound<'_, PyAny>> = o.try_iter()?.collect::<PyResult<_>>()?;
        let all = |f: &dyn Fn(&Bound<'_, PyAny>) -> bool| items.iter().all(f);
        if all(&|x| x.is_exact_instance_of::<PyBool>()) && !items.is_empty() {
            return Ok(Some(Lit::BoolList(items.iter().map(|x| x.extract()).collect::<PyResult<_>>()?)));
        }
        if all(&|x| x.is_exact_instance_of::<PyInt>()) {
            return Ok(Some(Lit::IntList(items.iter().map(|x| x.extract()).collect::<PyResult<_>>()?)));
        }
        if all(&|x| x.is_exact_instance_of::<PyFloat>()) {
            let fs: Vec<f64> = items.iter().map(|x| x.extract()).collect::<PyResult<_>>()?;
            return Ok(Some(Lit::FloatList(fs.into_iter().map(f64::to_bits).collect())));
        }
    }
    Ok(None)
}

// The op list as a verified Cfg of one block that leaves with `outputs`.
pub fn build(nparams: u32, ops: &Bound<'_, PyList>, outputs: &Bound<'_, PyList>) -> PyResult<Cfg> {
    let mut b = Builder::new();
    let params: Vec<ValueId> = (0..nparams).map(|i| b.param(i)).collect();
    b.vals.extend(&params);
    for op in ops.iter() {
        let (name, args, kwargs, nouts): (String, Bound<'_, PyTuple>, Bound<'_, PyDict>, Option<u32>) = op.extract()?;
        let mut vs: Vec<ValueId> = args.iter().map(|a| b.arg(&a)).collect::<PyResult<_>>()?;
        let mut kwnames = vec![];
        for (k, v) in kwargs.iter() {
            kwnames.push(k.extract::<String>()?);
            vs.push(b.arg(&v)?);
        }
        let v = b.add(Op::Aten { name, kwnames }, vs);
        match nouts {
            None => b.vals.push(v),
            Some(n) => {
                for k in 0..n {
                    let p = b.add(Op::Project(k), vec![v]);
                    b.vals.push(p);
                }
            }
        }
    }
    let outs: Vec<ValueId> = outputs.iter().map(|o| b.arg(&o)).collect::<PyResult<_>>()?;
    b.finish(params, outs)
}

/// Builds the graph of an op list (see graph.rs), prints it, and runs it on
/// `inputs`, returning (the printed graph, its outputs). For testing.
#[pyfunction]
pub fn _run_aten_graph<'py>(
    py: Python<'py>,
    nparams: u32,
    ops: &Bound<'py, PyList>,
    outputs: &Bound<'py, PyList>,
    inputs: Vec<Bound<'py, PyAny>>,
) -> PyResult<(String, Bound<'py, PyTuple>)> {
    let cfg = build(nparams, ops, outputs)?;
    let text = cfg.render(false, false);
    let prep = Prepared::new(Arc::new(cfg), None)?;
    let ins: Vec<*mut pyo3::ffi::PyObject> = inputs.iter().map(|i| i.as_ptr()).collect();
    let outs = unsafe { super::run::run_graph(&prep, &ins) }.map_err(|()| PyErr::fetch(py))?;
    let outs: Vec<Bound<'py, PyAny>> = outs.into_iter().map(|o| unsafe { Bound::from_owned_ptr(py, o) }).collect();
    Ok((text, PyTuple::new(py, outs)?))
}
