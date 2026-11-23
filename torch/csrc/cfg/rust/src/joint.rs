//! The joint graph of a form whose inputs need a gradient: the kernels of its
//! forward and, per set of results that come with a gradient, the operations
//! of its backward, as one straight-line graph over numbered values.
//!
//! It is taken down from a run on eager's tape (`backward.rs`):
//! the capture hands over every kernel the forward ran, with its operands and
//! what it gave, and then every kernel that eager's own autograd nodes ran
//! when they were called behind the forward, latest first. Which value an
//! operand is, is found by identity: the tensor itself, or, for a result that
//! autograd saved and wrapped anew, where its data is, its sizes and strides.
//! Two runs must come out as the same graph before the graph is used
//! (`differs`). `cut` decides what the forward keeps for the backward, and
//! `describe` hands the two plain graphs to what runs them.

use crate::capi::{self, TcfgTensorInfo};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyList, PyTuple};
use std::collections::{HashMap, HashSet};

pub type Value = u32;

/// Operators that look at what a tensor holds, or whose result has sizes
/// that depend on it: a graph with one of them is not the same at every
/// call, whatever two runs say.
const LOOKS: [&str; 22] = [
    "_local_scalar_dense",
    "item",
    "is_nonzero",
    "equal",
    "allclose",
    "nonzero",
    "nonzero_numpy",
    "nonzero_static",
    "argwhere",
    "masked_select",
    "unique",
    "_unique",
    "_unique2",
    "unique_dim",
    "unique_consecutive",
    "unique_dim_consecutive",
    "bincount",
    "_is_all_true",
    "_is_any_true",
    "tolist",
    "repeat_interleave",
    "count_nonzero_data",
];

/// Operators that a backward may run again in place of keeping what they
/// gave: each element of the result comes from the elements at its place
/// (or the result is a view), nothing is written and nothing is drawn.
const AGAIN: [&str; 64] = [
    "add", "sub", "rsub", "mul", "div", "neg", "abs", "relu", "sigmoid", "tanh", "exp", "log", "sqrt", "rsqrt",
    "reciprocal", "pow", "square", "clamp", "clamp_min", "clamp_max", "gelu", "silu", "hardtanh", "leaky_relu",
    "elu", "hardswish", "hardsigmoid", "softplus", "erf", "sin", "cos", "sign", "where", "eq", "ne", "lt", "le",
    "gt", "ge", "logical_not", "logical_and", "logical_or", "maximum", "minimum", "to", "_to_copy", "type_as",
    "view", "_unsafe_view", "reshape", "permute", "transpose", "t", "unsqueeze", "squeeze", "expand", "slice",
    "select", "flatten", "alias", "view_as", "expand_as", "contiguous", "detach",
];

enum Arg {
    Value(Value),
    Const(Py<PyAny>),
    /// A list, or with the flag a tuple, of arguments.
    Several(Vec<Arg>, bool),
}

struct Op {
    function: Py<PyAny>,
    name: String,
    args: Vec<Arg>,
    kwargs: Vec<(String, Arg)>,
    /// The tensors it gave, in order, and where each is in what it
    /// returned; `single` if what it returned is one tensor.
    results: Vec<Value>,
    places: Vec<u32>,
    single: bool,
    /// The values it wrote to, as they were in front of it.
    writes: Vec<Value>,
    random: bool,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct Layout {
    data: usize,
    sizes: Vec<i64>,
    strides: Vec<i64>,
    dtype: i32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Input(u32),
    Forward(u32),
    Incoming(u32),
    Backward,
}

struct Meta {
    kind: Kind,
    bytes: u64,
    /// Where the memory it is in starts: what is written to there changes it.
    storage: usize,
    /// Whether the caller of the function can get at it: an input, a result.
    reachable: bool,
    /// Eager's node that gave it and which of its results it is, for
    /// eager's text where it was written to before the backward; and which
    /// node of the backward that is, as the caller counts them.
    producer: Option<(String, u32)>,
    node: Option<u32>,
}

/// Who ran an operation of a backward: a node of eager's (its number, as the
/// caller counts them), `ADDS` where two gradients that meet are added, or
/// `NOBODY` where the caller did not say.
const ADDS: u32 = u32::MAX;
const NOBODY: u32 = u32::MAX - 1;

struct Back {
    pattern: Vec<bool>,
    /// The gradients that come in, one per result of the pattern.
    incoming: Vec<Value>,
    ops: Vec<Op>,
    owners: Vec<u32>,
    /// Per node of eager's: the gradients it was handed.
    handed: Vec<Vec<Value>>,
    /// The nodes that ran, in the order they ran.
    called: Vec<u32>,
    /// Per contribution, in eager's order: which one it is (the caller
    /// numbers them over all backwards), and the value.
    gives: Vec<(u32, Value)>,
}

#[pyclass(module = "torch._cfg_rs", name = "Joint")]
pub struct Joint {
    inputs: u32,
    forward: Vec<Op>,
    results: Vec<Value>,
    needs: Vec<bool>,
    backs: Vec<Back>,
    open: bool,
    metas: Vec<Meta>,
    by_object: HashMap<usize, Value>,
    by_layout: HashMap<Layout, Value>,
    /// The tensors of the run, kept while it is taken down: one that died
    /// could give its place to another.
    kept: Vec<Py<PyAny>>,
    tensor: Py<PyAny>,
    saved: Vec<Value>,
    /// Who runs the operations of the backward that is open, and who did in
    /// front of an addition.
    owner: u32,
    before: u32,
    /// Per node of eager's: the values of the forward it kept, in the order
    /// it looks at them when it runs.
    unpacks: Vec<Vec<Value>>,
    /// Per node of eager's, per edge that what it gives goes along: the
    /// node it leads to, or the contribution it is.
    leads: Vec<Vec<(Option<u32>, Option<u32>)>>,
}

/// A backward for some of the results, made of one that was taken down.
struct Part<'a> {
    back: &'a Back,
    pattern: Vec<bool>,
    incoming: Vec<Value>,
    ops: Vec<usize>,
    /// A value that is another one here: a sum with one side missing.
    alias: HashMap<Value, Value>,
    gives: Vec<(u32, Value)>,
    /// The nodes of eager's that run, in the order they run.
    nodes: Vec<u32>,
}

impl<'a> Part<'a> {
    fn whole(back: &'a Back) -> Self {
        Part {
            back,
            pattern: back.pattern.clone(),
            incoming: back.incoming.clone(),
            ops: (0..back.ops.len()).collect(),
            alias: HashMap::new(),
            gives: back.gives.clone(),
            nodes: back.called.clone(),
        }
    }
}

fn item_size(dtype: i32) -> u64 {
    // Byte, Char, Short, Int, Long, Half, Float, Double, ComplexHalf,
    // ComplexFloat, ComplexDouble, Bool, QInt8, QUInt8, QInt32, BFloat16.
    const SIZES: [u64; 16] = [1, 1, 2, 4, 8, 2, 4, 8, 4, 8, 16, 1, 1, 1, 4, 2];
    SIZES.get(dtype as usize).copied().unwrap_or(8)
}

fn refused(why: String) -> PyErr {
    PyValueError::new_err(why)
}

impl Joint {
    /// What the engine knows of a tensor; None for what is none it knows (a
    /// subclass, a sparse one).
    fn info(&self, obj: &Bound<'_, PyAny>) -> Option<(Layout, u64, usize)> {
        let api = capi::api()?;
        let mut info = TcfgTensorInfo::empty();
        if unsafe { (api.tensor_info)(obj.as_ptr(), &mut info) } == 0 || info.strides.is_null() {
            return None;
        }
        let n = info.ndim as usize;
        let sizes = unsafe { std::slice::from_raw_parts(info.sizes, n) }.to_vec();
        let strides = unsafe { std::slice::from_raw_parts(info.strides, n) }.to_vec();
        let count: i64 = sizes.iter().product();
        let size = item_size(info.dtype);
        let storage = (info.data as usize).wrapping_sub(info.storage_offset as usize * size as usize);
        Some((Layout { data: info.data as usize, sizes, strides, dtype: info.dtype }, count as u64 * size, storage))
    }

    fn is_tensor(&self, obj: &Bound<'_, PyAny>) -> bool {
        obj.is_instance(self.tensor.bind(obj.py())).unwrap_or(false)
    }

    fn find(&self, obj: &Bound<'_, PyAny>) -> Option<Value> {
        if let Some(found) = self.by_object.get(&(obj.as_ptr() as usize)) {
            return Some(*found);
        }
        let (layout, bytes, _) = self.info(obj)?;
        if bytes == 0 {
            return None;
        }
        self.by_layout.get(&layout).copied()
    }

    /// A new value for a tensor: what the tensor is from here on.
    fn define(&mut self, obj: &Bound<'_, PyAny>, kind: Kind, what: &str) -> PyResult<Value> {
        let Some((layout, bytes, storage)) = self.info(obj) else {
            return Err(refused(format!("{what} is a tensor the engine does not know (a subclass, or one that is not dense)")));
        };
        let value = self.metas.len() as Value;
        self.metas.push(Meta { kind, bytes, storage, reachable: false, producer: None, node: None });
        // What is found by where it is, is the tensor that was there first
        // (a view with the sizes of what it is a view of is in the same
        // place), as an operation that wrote to it left it.
        let written = self.by_object.insert(obj.as_ptr() as usize, value).is_some();
        if bytes > 0 && (written || !self.by_layout.contains_key(&layout)) {
            self.by_layout.insert(layout, value);
        }
        self.kept.push(obj.clone().unbind());
        Ok(value)
    }

    fn arg(&self, obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Arg> {
        if self.is_tensor(obj) {
            return match self.find(obj) {
                Some(value) => Ok(Arg::Value(value)),
                None => Err(refused(format!("an operand of {name} is no value of the graph"))),
            };
        }
        let tuple = obj.is_instance_of::<PyTuple>();
        if tuple || obj.is_instance_of::<PyList>() {
            let items: PyResult<Vec<Arg>> = obj.try_iter()?.map(|item| self.arg(&item?, name)).collect();
            return Ok(Arg::Several(items?, tuple));
        }
        Ok(Arg::Const(obj.clone().unbind()))
    }

    #[allow(clippy::too_many_arguments)]
    fn take(
        &mut self,
        function: &Bound<'_, PyAny>,
        name: &str,
        args: &Bound<'_, PyTuple>,
        kwargs: &Bound<'_, PyDict>,
        out: &Bound<'_, PyAny>,
        written: &[Bound<'_, PyAny>],
        random: bool,
        kind: Kind,
    ) -> PyResult<Op> {
        let base = name.split("::").last().unwrap_or(name).split('.').next().unwrap_or(name).trim_end_matches('_');
        if LOOKS.contains(&base) {
            return Err(refused(format!("{name} looks at what a tensor holds")));
        }
        let taken: PyResult<Vec<Arg>> = args.iter().map(|item| self.arg(&item, name)).collect();
        let mut named = Vec::new();
        for (key, item) in kwargs.iter() {
            named.push((key.extract::<String>()?, self.arg(&item, name)?));
        }
        let writes: Vec<Value> = written.iter().filter_map(|tensor| self.find(tensor)).collect();
        let single = self.is_tensor(out);
        let (mut results, mut places) = (Vec::new(), Vec::new());
        if single {
            results.push(self.define(out, kind, name)?);
            places.push(0);
        } else if out.is_instance_of::<PyTuple>() || out.is_instance_of::<PyList>() {
            // What is no tensor among what it gives (a number that says
            // which kernel ran) is not read by anything.
            for (at, item) in out.try_iter()?.enumerate() {
                let item = item?;
                if self.is_tensor(&item) {
                    results.push(self.define(&item, kind, name)?);
                    places.push(at as u32);
                }
            }
        } else if !out.is_none() {
            return Err(refused(format!("{name} gives what is no tensor")));
        }
        let function = function.clone().unbind();
        Ok(Op { function, name: name.to_string(), args: taken?, kwargs: named, results, places, single, writes, random })
    }

    fn back(&mut self) -> PyResult<&mut Back> {
        if !self.open {
            return Err(PyRuntimeError::new_err("no backward is being taken down"));
        }
        Ok(self.backs.last_mut().expect("one is open"))
    }

    fn same_arg(py: Python<'_>, a: &Arg, b: &Arg) -> bool {
        match (a, b) {
            (Arg::Value(a), Arg::Value(b)) => a == b,
            (Arg::Several(a, s), Arg::Several(b, t)) => {
                s == t && a.len() == b.len() && a.iter().zip(b).all(|(a, b)| Self::same_arg(py, a, b))
            }
            (Arg::Const(a), Arg::Const(b)) => {
                let (a, b) = (a.bind(py), b.bind(py));
                let nan = |o: &Bound<'_, PyAny>| o.cast::<PyFloat>().is_ok_and(|f| f.value().is_nan());
                a.get_type().is(&b.get_type()) && (a.is(b) || (nan(a) && nan(b)) || a.eq(b).unwrap_or(false))
            }
            _ => false,
        }
    }

    fn same_ops(py: Python<'_>, a: &[Op], b: &[Op], what: &str) -> Option<String> {
        if a.len() != b.len() {
            return Some(format!("{what} has {} operations in one run and {} in another", a.len(), b.len()));
        }
        for (at, (a, b)) in a.iter().zip(b).enumerate() {
            let alike = a.name == b.name
                && a.results == b.results
                && a.places == b.places
                && a.single == b.single
                && a.writes == b.writes
                && a.args.len() == b.args.len()
                && a.args.iter().zip(&b.args).all(|(a, b)| Self::same_arg(py, a, b))
                && a.kwargs.len() == b.kwargs.len()
                && a.kwargs.iter().zip(&b.kwargs).all(|((k, a), (l, b))| k == l && Self::same_arg(py, a, b));
            if !alike {
                return Some(format!("operation {at} of {what} ({}) is not the same in two runs", a.name));
            }
        }
        None
    }

    fn described_arg<'py>(py: Python<'py>, arg: &Arg, alias: &HashMap<Value, Value>) -> PyResult<Bound<'py, PyAny>> {
        Ok(match arg {
            Arg::Value(value) => ("v", *alias.get(value).unwrap_or(value)).into_pyobject(py)?.into_any(),
            Arg::Const(held) => ("c", held.bind(py)).into_pyobject(py)?.into_any(),
            Arg::Several(items, tuple) => {
                let items: PyResult<Vec<Bound<'py, PyAny>>> = items.iter().map(|item| Self::described_arg(py, item, alias)).collect();
                (if *tuple { "t" } else { "l" }, PyList::new(py, items?)?).into_pyobject(py)?.into_any()
            }
        })
    }

    fn described_op<'py>(py: Python<'py>, op: &Op, alias: &HashMap<Value, Value>) -> PyResult<Bound<'py, PyAny>> {
        let args: PyResult<Vec<Bound<'py, PyAny>>> = op.args.iter().map(|arg| Self::described_arg(py, arg, alias)).collect();
        let kwargs = PyDict::new(py);
        for (key, arg) in &op.kwargs {
            kwargs.set_item(key, Self::described_arg(py, arg, alias)?)?;
        }
        let results: Vec<(u32, Value)> = op.places.iter().copied().zip(op.results.iter().copied()).collect();
        Ok((op.function.bind(py), PyList::new(py, args?)?, kwargs, results, op.single, op.name.as_str()).into_pyobject(py)?.into_any())
    }

    /// The forward and input values an operation reads.
    fn reads(&self, op: &Op, into: &mut Vec<Value>) {
        fn walk(arg: &Arg, into: &mut Vec<Value>) {
            match arg {
                Arg::Value(value) => into.push(*value),
                Arg::Several(items, _) => items.iter().for_each(|item| walk(item, into)),
                Arg::Const(_) => {}
            }
        }
        let mut all = Vec::new();
        op.args.iter().for_each(|arg| walk(arg, &mut all));
        op.kwargs.iter().for_each(|(_, arg)| walk(arg, &mut all));
        into.extend(all.into_iter().filter(|v| matches!(self.metas[*v as usize].kind, Kind::Input(_) | Kind::Forward(_))));
    }

    /// The backward for the results of `pattern`. One that was taken down
    /// for just these is given as it is. Else it is made of one that was
    /// taken down for more: a gradient that does not come in is missing; a
    /// node of eager's that is handed nothing does not run and gives
    /// nothing (the caller has seen that it gives nothing then); where two
    /// gradients meet and one is missing, the other is the sum. A node
    /// that would be handed some of its gradients and not others was not
    /// taken down like that, and is refused; with `lenient` it runs, for
    /// what only asks which nodes run. With `needed` (per contribution,
    /// whether it is asked for) the nodes that lead to none that is asked
    /// for do not run, as the autograd engine runs a backward for some
    /// inputs: a node that runs does all it does.
    fn part(&self, pattern: &[bool], lenient: bool, needed: Option<&[bool]>) -> PyResult<Part<'_>> {
        let mut part = self.part_of(pattern, lenient)?;
        let Some(needed) = needed else {
            return Ok(part);
        };
        let back = part.back;
        let need = self.asked(needed);
        let runs = |owner: u32| need.get(owner as usize).copied().unwrap_or(true);
        part.nodes.retain(|node| runs(*node));
        part.gives.retain(|(slot, _)| needed.get(*slot as usize).copied().unwrap_or(true));
        // Where gradients meet, the sum is made if a node that runs is
        // handed it, or it is what is given.
        let mut live: HashSet<Value> = part.gives.iter().map(|(_, value)| *value).collect();
        for node in &part.nodes {
            let handed = back.handed.get(*node as usize).into_iter().flatten();
            live.extend(handed.map(|v| *part.alias.get(v).unwrap_or(v)));
        }
        let mut kept: Vec<usize> = Vec::new();
        for at in part.ops.iter().rev() {
            let (owner, op) = (back.owners[*at], &back.ops[*at]);
            let keep = match owner {
                ADDS => op.results.iter().any(|v| live.contains(v)),
                NOBODY => true,
                node => runs(node),
            };
            if keep {
                if owner == ADDS {
                    let mut sides = Vec::new();
                    self.sides(op, &mut sides);
                    live.extend(sides.iter().map(|v| *part.alias.get(v).unwrap_or(v)));
                }
                kept.push(*at);
            }
        }
        kept.reverse();
        part.ops = kept;
        Ok(part)
    }

    /// The tensors an operation takes: values of the forward and of the
    /// backward.
    fn sides(&self, op: &Op, into: &mut Vec<Value>) {
        fn walk(arg: &Arg, into: &mut Vec<Value>) {
            match arg {
                Arg::Value(value) => into.push(*value),
                Arg::Several(items, _) => items.iter().for_each(|item| walk(item, into)),
                Arg::Const(_) => {}
            }
        }
        op.args.iter().for_each(|arg| walk(arg, into));
        op.kwargs.iter().for_each(|(_, arg)| walk(arg, into));
    }

    /// Which nodes of eager's lead to a contribution that is asked for.
    fn asked(&self, needed: &[bool]) -> Vec<bool> {
        let wanted = |slot: u32| needed.get(slot as usize).copied().unwrap_or(true);
        let mut need = vec![false; self.leads.len()];
        loop {
            let mut changed = false;
            // A node leads to nodes that were made before it, which come
            // behind it here.
            for node in (0..self.leads.len()).rev() {
                let leads = |(next, slot): &(Option<u32>, Option<u32>)| {
                    slot.is_some_and(wanted) || next.is_some_and(|next| need.get(next as usize).copied().unwrap_or(true))
                };
                if !need[node] && self.leads[node].iter().any(leads) {
                    need[node] = true;
                    changed = true;
                }
            }
            if !changed {
                return need;
            }
        }
    }

    fn part_of(&self, pattern: &[bool], lenient: bool) -> PyResult<Part<'_>> {
        if pattern.len() != self.results.len() || !pattern.iter().any(|on| *on) {
            return Err(refused("a backward is for some of the results of the graph".to_string()));
        }
        if let Some(back) = self.backs.iter().find(|back| back.pattern == pattern) {
            return Ok(Part::whole(back));
        }
        let covers = |back: &&Back| pattern.iter().zip(&back.pattern).all(|(on, had)| !*on || *had);
        let Some(back) = self.backs.iter().find(covers) else {
            return Err(refused("no backward was taken down for these results".to_string()));
        };
        let mut part = Part::whole(back);
        part.pattern = pattern.to_vec();
        part.ops.clear();
        part.gives.clear();
        part.incoming.clear();
        part.nodes.clear();
        let mut missing: HashSet<Value> = HashSet::new();
        let had = (0..pattern.len()).filter(|at| back.pattern[*at]);
        for (at, value) in had.zip(&back.incoming) {
            if pattern[at] {
                part.incoming.push(*value);
            } else {
                missing.insert(*value);
            }
        }
        let partly = || refused("a node of eager's would be handed some of its gradients and not others, which was not taken down".to_string());
        // Whether a node runs here: it is handed what it was handed.
        let handed = |part: &Part<'_>, missing: &HashSet<Value>, owner: u32| -> PyResult<bool> {
            let handed = back.handed.get(owner as usize).ok_or_else(partly)?;
            let gone = handed.iter().filter(|v| missing.contains(part.alias.get(*v).unwrap_or(*v))).count();
            if gone != 0 && gone != handed.len() && !lenient {
                return Err(partly());
            }
            Ok(gone == 0 || gone != handed.len())
        };
        let mut runs: HashMap<u32, bool> = HashMap::new();
        for (at, op) in back.ops.iter().enumerate() {
            let owner = back.owners[at];
            if owner == ADDS {
                let sides: Vec<Value> = op.args.iter().filter_map(|arg| if let Arg::Value(v) = arg { Some(*v) } else { None }).collect();
                // What else the addition takes (a factor of one) is no side.
                if sides.len() != 2 || !op.single {
                    if lenient {
                        part.ops.push(at);
                        continue;
                    }
                    return Err(refused("where two gradients meet is not one addition".to_string()));
                }
                let sides: Vec<Value> = sides.iter().map(|v| *part.alias.get(v).unwrap_or(v)).collect();
                match (missing.contains(&sides[0]), missing.contains(&sides[1])) {
                    (true, true) => {
                        missing.insert(op.results[0]);
                    }
                    (true, false) => {
                        part.alias.insert(op.results[0], sides[1]);
                    }
                    (false, true) => {
                        part.alias.insert(op.results[0], sides[0]);
                    }
                    (false, false) => part.ops.push(at),
                }
                continue;
            }
            if owner == NOBODY {
                if lenient {
                    part.ops.push(at);
                    continue;
                }
                return Err(refused("no backward was taken down for these results".to_string()));
            }
            let run = match runs.get(&owner) {
                Some(run) => *run,
                None => {
                    let run = handed(&part, &missing, owner)?;
                    runs.insert(owner, run);
                    run
                }
            };
            if run {
                part.ops.push(at);
            } else {
                missing.extend(op.results.iter().copied());
            }
        }
        // A node that ran no operation runs like the others: if it is
        // handed what it was handed.
        for owner in &back.called {
            let run = match runs.get(owner) {
                Some(run) => *run,
                None => handed(&part, &missing, *owner)?,
            };
            if run {
                part.nodes.push(*owner);
            }
        }
        for (slot, value) in &back.gives {
            let value = *part.alias.get(value).unwrap_or(value);
            if !missing.contains(&value) {
                part.gives.push((*slot, value));
            }
        }
        Ok(part)
    }

    /// The forward operations a backward runs again, in their order, for
    /// what it reads and is not kept.
    fn again_for(&self, part: &Part<'_>) -> Vec<u32> {
        let kept: HashSet<Value> = self.saved.iter().copied().collect();
        let mut wanted: Vec<Value> = Vec::new();
        part.ops.iter().for_each(|at| self.reads(&part.back.ops[*at], &mut wanted));
        wanted.extend(part.gives.iter().map(|(_, value)| *value));
        let mut again: Vec<u32> = Vec::new();
        while let Some(v) = wanted.pop() {
            if kept.contains(&v) {
                continue;
            }
            if let Kind::Forward(op) = self.metas[v as usize].kind {
                if !again.contains(&op) {
                    again.push(op);
                    self.reads(&self.forward[op as usize], &mut wanted);
                }
            }
        }
        again.sort_unstable();
        again
    }

    fn described_part<'py>(&self, py: Python<'py>, part: &Part<'_>) -> PyResult<Bound<'py, PyAny>> {
        let none = HashMap::new();
        let mut all: Vec<Bound<'py, PyAny>> = Vec::new();
        for at in self.again_for(part) {
            all.push(Self::described_op(py, &self.forward[at as usize], &none)?);
        }
        for at in &part.ops {
            all.push(Self::described_op(py, &part.back.ops[*at], &part.alias)?);
        }
        Ok((part.pattern.clone(), part.incoming.clone(), PyList::new(py, all)?, part.gives.clone()).into_pyobject(py)?.into_any())
    }

    /// How many operations the forward has.
    pub(crate) fn forward_len(&self) -> usize {
        self.forward.len()
    }

    /// The values an operation of the forward takes, and those it gives.
    pub(crate) fn forward_reads(&self, at: usize) -> Vec<Value> {
        let mut taken = Vec::new();
        self.reads(&self.forward[at], &mut taken);
        taken
    }

    pub(crate) fn forward_gives(&self, at: usize) -> &[Value] {
        &self.forward[at].results
    }

    /// Operations of the forward, described as `describe` describes them.
    pub(crate) fn described_forward<'py>(&self, py: Python<'py>, which: &[usize]) -> PyResult<Bound<'py, PyList>> {
        let none = HashMap::new();
        let all: PyResult<Vec<Bound<'py, PyAny>>> = which.iter().map(|at| Self::described_op(py, &self.forward[*at], &none)).collect();
        PyList::new(py, all?)
    }

    /// How many values the graph has.
    pub(crate) fn values(&self) -> u32 {
        self.metas.len() as u32
    }

    /// Eager's node for a value and which of its results the value is; with
    /// `node`, which node of the backward it is, as the caller counts them.
    pub(crate) fn made_by(&mut self, value: Value, name: String, output: u32, node: Option<u32>) {
        let meta = &mut self.metas[value as usize];
        meta.producer = Some((name, output));
        meta.node = node;
    }

    /// The values of the forward that the backwards read.
    pub(crate) fn read(&self) -> Vec<Value> {
        let mut read: Vec<Value> = Vec::new();
        for back in &self.backs {
            back.ops.iter().for_each(|op| self.reads(op, &mut read));
            read.extend(back.gives.iter().map(|(_, value)| *value));
        }
        read.retain(|v| matches!(self.metas[*v as usize].kind, Kind::Input(_) | Kind::Forward(_)));
        read.sort_unstable();
        read.dedup();
        read
    }

    /// The memory the caller of the function can get at: where the inputs
    /// and the results are.
    fn outside(&self) -> HashSet<usize> {
        self.metas.iter().filter(|meta| meta.reachable).map(|meta| meta.storage).collect()
    }

    /// How many values the form gives.
    pub(crate) fn results_len(&self) -> usize {
        self.results.len()
    }

    /// The backward for the results of `pattern` and, with `needed`, for
    /// the contributions that are asked for, for what runs it: its
    /// operations (described), the values of the gradients that come in,
    /// what it gives, and the values of the forward it takes.
    #[allow(clippy::type_complexity)]
    pub(crate) fn part_program<'py>(
        &self,
        py: Python<'py>,
        pattern: &[bool],
        needed: Option<&[bool]>,
    ) -> PyResult<(Bound<'py, PyList>, Vec<Value>, Vec<(u32, Value)>, Vec<Value>)> {
        let part = self.part(pattern, false, needed)?;
        let again = self.again_for(&part);
        let none = HashMap::new();
        let mut all: Vec<Bound<'py, PyAny>> = Vec::new();
        let mut made: HashSet<Value> = HashSet::new();
        let mut takes: Vec<Value> = Vec::new();
        let mut read = |op: &Op, made: &HashSet<Value>| {
            let mut found = Vec::new();
            self.reads(op, &mut found);
            for value in found {
                if !made.contains(&value) && !takes.contains(&value) {
                    takes.push(value);
                }
            }
        };
        for at in &again {
            let op = &self.forward[*at as usize];
            read(op, &made);
            made.extend(op.results.iter().copied());
            all.push(Self::described_op(py, op, &none)?);
        }
        for at in &part.ops {
            let op = &part.back.ops[*at];
            read(op, &made);
            all.push(Self::described_op(py, op, &part.alias)?);
        }
        for (_, value) in &part.gives {
            let forward = matches!(self.metas[*value as usize].kind, Kind::Input(_) | Kind::Forward(_));
            if forward && !made.contains(value) && !takes.contains(value) {
                takes.push(*value);
            }
        }
        Ok((PyList::new(py, all)?, part.incoming.clone(), part.gives.clone(), takes))
    }

    /// What a node of eager's kept of the forward, in the order it looks at
    /// it when it runs. Asked for after `made_by` said which node gave
    /// what: a node looks at what it kept of its operands first.
    pub(crate) fn kept_by(&mut self, node: u32, values: Vec<Value>) {
        let forward = |v: &&Value| matches!(self.metas[**v as usize].kind, Kind::Input(_) | Kind::Forward(_));
        let own = |v: &&Value| self.metas[**v as usize].node == Some(node);
        let mut kept: Vec<Value> = values.iter().filter(forward).filter(|v| !own(v)).copied().collect();
        kept.extend(values.iter().filter(forward).filter(own));
        if self.unpacks.len() <= node as usize {
            self.unpacks.resize(node as usize + 1, Vec::new());
        }
        self.unpacks[node as usize] = kept;
    }

    /// Where what a node of eager's gives goes, per edge: the node (as the
    /// caller counts them) or the contribution.
    pub(crate) fn edges(&mut self, node: u32, leads: Vec<(Option<u32>, Option<u32>)>) {
        if self.leads.len() <= node as usize {
            self.leads.resize(node as usize + 1, Vec::new());
        }
        self.leads[node as usize] = leads;
    }

    /// The values of the forward that eager's nodes look at when a backward
    /// for the results of `pattern` (and the contributions of `needed`)
    /// runs, in the order they do, each with whether the node that looks
    /// at it first is the one that gave it.
    pub(crate) fn looked_at(&self, pattern: &[bool], needed: Option<&[bool]>) -> Vec<(Value, bool)> {
        let Ok(part) = self.part(pattern, true, needed) else {
            return Vec::new();
        };
        let mut reads: Vec<(Value, bool)> = Vec::new();
        let mut add = |value: Value, node: Option<u32>| {
            if !reads.iter().any(|(known, _)| *known == value) {
                reads.push((value, node.is_some() && self.metas[value as usize].node == node));
            }
        };
        for node in &part.nodes {
            for value in self.unpacks.get(*node as usize).into_iter().flatten() {
                add(*value, Some(*node));
            }
            // What a node reads and does not say it kept.
            let mut found = Vec::new();
            for at in part.ops.iter().filter(|at| part.back.owners[**at] == *node) {
                self.reads(&part.back.ops[*at], &mut found);
            }
            found.iter().for_each(|value| add(*value, Some(*node)));
        }
        let mut found = Vec::new();
        for at in part.ops.iter().filter(|at| !part.nodes.contains(&part.back.owners[**at])) {
            self.reads(&part.back.ops[*at], &mut found);
        }
        found.extend(part.gives.iter().map(|(_, value)| *value).filter(|v| matches!(self.metas[*v as usize].kind, Kind::Input(_) | Kind::Forward(_))));
        found.iter().for_each(|value| add(*value, None));
        reads
    }

    /// Whether a backward may run a forward operation again: it is one of
    /// `AGAIN`, gives one tensor, takes nothing in memory the caller can
    /// get at (`outside`: what is written there behind the forward is not
    /// what the forward read), and nothing behind it in the forward wrote
    /// to the memory of what it takes.
    fn runs_again(&self, at: usize, outside: &HashSet<usize>) -> bool {
        let op = &self.forward[at];
        let base = op.name.split("::").last().unwrap_or("").split('.').next().unwrap_or("");
        if !AGAIN.contains(&base) || !op.single || !op.writes.is_empty() || op.random {
            return false;
        }
        let mut taken = Vec::new();
        self.reads(op, &mut taken);
        let storages: HashSet<usize> = taken.iter().map(|v| self.metas[*v as usize].storage).collect();
        if storages.iter().any(|storage| outside.contains(storage)) {
            return false;
        }
        let written = |later: &Op| later.writes.iter().any(|v| storages.contains(&self.metas[*v as usize].storage));
        !self.forward[at + 1..].iter().any(written)
    }
}

/// The smallest set of edges, by capacity, that parts the first node from
/// the second: which nodes are still reached from the first (Edmonds and
/// Karp; the graphs are those of one form).
fn min_cut(n: usize, edges: &[(usize, usize, u64)]) -> Vec<bool> {
    let mut to: Vec<usize> = Vec::new();
    let mut cap: Vec<u64> = Vec::new();
    let mut out: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (a, b, c) in edges {
        out[*a].push(to.len());
        to.push(*b);
        cap.push(*c);
        out[*b].push(to.len());
        to.push(*a);
        cap.push(0);
    }
    loop {
        let mut from: Vec<Option<usize>> = vec![None; n];
        let mut queue = std::collections::VecDeque::from([0usize]);
        let mut seen = vec![false; n];
        seen[0] = true;
        while let Some(at) = queue.pop_front() {
            for e in &out[at] {
                if cap[*e] > 0 && !seen[to[*e]] {
                    seen[to[*e]] = true;
                    from[to[*e]] = Some(*e);
                    queue.push_back(to[*e]);
                }
            }
        }
        if !seen[1] {
            return seen;
        }
        let (mut at, mut least) = (1, u64::MAX);
        while let Some(e) = from[at] {
            least = least.min(cap[e]);
            at = to[e ^ 1];
        }
        at = 1;
        while let Some(e) = from[at] {
            cap[e] -= least;
            cap[e ^ 1] += least;
            at = to[e ^ 1];
        }
    }
}

#[pymethods]
impl Joint {
    /// A graph over the tensors a form is handed, in their order.
    #[new]
    pub fn new(py: Python<'_>, inputs: Vec<Bound<'_, PyAny>>) -> PyResult<Self> {
        let tensor = py.import("torch")?.getattr("Tensor")?.unbind();
        let mut joint = Joint {
            inputs: inputs.len() as u32,
            forward: Vec::new(),
            results: Vec::new(),
            needs: Vec::new(),
            backs: Vec::new(),
            open: false,
            metas: Vec::new(),
            by_object: HashMap::new(),
            by_layout: HashMap::new(),
            kept: Vec::new(),
            tensor,
            saved: Vec::new(),
            owner: NOBODY,
            before: NOBODY,
            unpacks: Vec::new(),
            leads: Vec::new(),
        };
        for (at, input) in inputs.iter().enumerate() {
            let value = joint.define(input, Kind::Input(at as u32), "an input")?;
            joint.metas[value as usize].reachable = true;
        }
        Ok(joint)
    }

    /// A kernel the forward ran: the operator, what it was called with, what
    /// it returned, the operands it wrote to, and whether it draws.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        function: &Bound<'_, PyAny>,
        name: &str,
        args: &Bound<'_, PyTuple>,
        kwargs: &Bound<'_, PyDict>,
        out: &Bound<'_, PyAny>,
        written: Vec<Bound<'_, PyAny>>,
        random: bool,
    ) -> PyResult<()> {
        let at = self.forward.len() as u32;
        let op = self.take(function, name, args, kwargs, out, &written, random, Kind::Forward(at))?;
        self.forward.push(op);
        Ok(())
    }

    /// The tensors the form gives, and which of them need a gradient.
    pub fn results(&mut self, tensors: Vec<Bound<'_, PyAny>>, needs: Vec<bool>) -> PyResult<()> {
        for tensor in &tensors {
            let Some(value) = self.find(tensor) else {
                return Err(refused("a result is no value of the graph".to_string()));
            };
            self.metas[value as usize].reachable = true;
            self.results.push(value);
        }
        self.needs = needs;
        Ok(())
    }

    /// Which value of the graph a tensor is, while the run is taken down.
    pub fn value(&self, tensor: &Bound<'_, PyAny>) -> Option<Value> {
        self.find(tensor)
    }

    /// Starts the backward for the results that come with a gradient
    /// (`pattern`, one flag per result), with those gradients.
    pub fn begin(&mut self, pattern: Vec<bool>, incoming: Vec<Bound<'_, PyAny>>) -> PyResult<()> {
        if self.open || pattern.len() != self.results.len() || incoming.len() != pattern.iter().filter(|p| **p).count() {
            return Err(PyRuntimeError::new_err("a backward is begun for the results of the graph, one at a time"));
        }
        let which: Vec<u32> = (0..pattern.len() as u32).filter(|at| pattern[*at as usize]).collect();
        let mut values = Vec::new();
        for (at, tensor) in which.iter().zip(&incoming) {
            values.push(self.define(tensor, Kind::Incoming(*at), "an incoming gradient")?);
        }
        let (ops, owners, handed, gives) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        self.backs.push(Back { pattern, incoming: values, ops, owners, handed, called: Vec::new(), gives });
        self.open = true;
        (self.owner, self.before) = (NOBODY, NOBODY);
        Ok(())
    }

    /// The operations from here on are run by a node of eager's (`index`,
    /// as the caller counts them), which was handed these gradients.
    pub fn node(&mut self, index: u32, handed: Vec<Bound<'_, PyAny>>) -> PyResult<()> {
        let mut values = Vec::new();
        for tensor in &handed {
            values.push(self.find(tensor).ok_or_else(|| refused("a gradient a node is handed is no value of the graph".to_string()))?);
        }
        let back = self.back()?;
        if back.handed.len() <= index as usize {
            back.handed.resize(index as usize + 1, Vec::new());
        }
        back.handed[index as usize] = values;
        back.called.push(index);
        self.owner = index;
        Ok(())
    }

    /// With `on`, the operation that follows adds two gradients that meet;
    /// without, the node runs again.
    pub fn adds(&mut self, on: bool) {
        if on {
            (self.before, self.owner) = (self.owner, ADDS);
        } else {
            self.owner = self.before;
        }
    }

    /// The backward for the results of `pattern`, as `describe` gives one.
    pub fn backward_for<'py>(&self, py: Python<'py>, pattern: Vec<bool>) -> PyResult<Bound<'py, PyAny>> {
        self.described_part(py, &self.part(&pattern, false, None)?)
    }

    /// The tensor of a value, while the run is taken down.
    pub fn tensor<'py>(&self, py: Python<'py>, value: Value) -> Option<Bound<'py, PyAny>> {
        self.kept.get(value as usize).map(|held| held.bind(py).clone())
    }

    /// A kernel a node of eager's ran for the backward that is open.
    pub fn backward(
        &mut self,
        function: &Bound<'_, PyAny>,
        name: &str,
        args: &Bound<'_, PyTuple>,
        kwargs: &Bound<'_, PyDict>,
        out: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.back()?;
        let op = self.take(function, name, args, kwargs, out, &[], false, Kind::Backward)?;
        let owner = self.owner;
        let back = self.back()?;
        back.ops.push(op);
        back.owners.push(owner);
        Ok(())
    }

    /// What a node gave for the gradient of an input: which contribution it
    /// is, and the tensor.
    pub fn give(&mut self, input: u32, tensor: &Bound<'_, PyAny>) -> PyResult<()> {
        let value = self.find(tensor).ok_or_else(|| refused("what a node gives is no value of the graph".to_string()))?;
        self.back()?.gives.push((input, value));
        Ok(())
    }

    pub fn end(&mut self) -> PyResult<()> {
        self.back()?;
        self.open = false;
        Ok(())
    }

    /// The run is taken down: its tensors are let go.
    pub fn finish(&mut self) {
        self.by_object.clear();
        self.by_layout.clear();
        self.kept.clear();
    }

    /// None if another run was taken down as the same graph; else what is
    /// not the same.
    pub fn differs(&self, py: Python<'_>, other: &Joint) -> Option<String> {
        if self.inputs != other.inputs || self.results != other.results || self.needs != other.needs || self.leads != other.leads {
            return Some("the inputs or the results are not the same in two runs".to_string());
        }
        if let Some(why) = Self::same_ops(py, &self.forward, &other.forward, "the forward") {
            return Some(why);
        }
        if self.backs.len() != other.backs.len() {
            return Some("not the same backwards in two runs".to_string());
        }
        for (a, b) in self.backs.iter().zip(&other.backs) {
            if a.pattern != b.pattern || a.gives != b.gives || a.incoming != b.incoming || a.owners != b.owners || a.handed != b.handed {
                return Some("a backward gives other gradients in another run".to_string());
            }
            if let Some(why) = Self::same_ops(py, &a.ops, &b.ops, "a backward") {
                return Some(why);
            }
        }
        None
    }

    /// Decides what the forward keeps for the backwards. Without `least`:
    /// every forward value a backward reads. With it: the set of the fewest
    /// bytes from which the backwards can compute what they read, by
    /// running forward operations again (`runs_again`). What is in memory
    /// the caller can get at is kept if and only if a backward reads it,
    /// as eager keeps it: it is what is checked for writes at the
    /// backward. Returns the bytes of what the backwards read and of what
    /// is kept.
    pub fn cut(&mut self, least: bool) -> (u64, u64) {
        let read = self.read();
        let outside = self.outside();
        let bytes = |values: &[Value]| values.iter().map(|v| self.metas[*v as usize].bytes).sum::<u64>();
        let all = bytes(&read);
        if !least {
            self.saved = read;
            return (all, all);
        }
        // Per forward value an edge of its bytes, from where it comes in to
        // where it goes out: cutting it keeps the value.
        const NEVER: u64 = u64::MAX / 4;
        let count = self.metas.iter().filter(|meta| !matches!(meta.kind, Kind::Incoming(_) | Kind::Backward)).count();
        let forward: Vec<Value> =
            (0..self.metas.len() as Value).filter(|v| matches!(self.metas[*v as usize].kind, Kind::Input(_) | Kind::Forward(_))).collect();
        let place: HashMap<Value, usize> = forward.iter().enumerate().map(|(at, v)| (*v, at)).collect();
        let (into, outof) = (|v: Value| 2 + 2 * place[&v], |v: Value| 3 + 2 * place[&v]);
        let mut edges: Vec<(usize, usize, u64)> = Vec::new();
        let reads: HashSet<Value> = read.iter().copied().collect();
        for v in &forward {
            let meta = &self.metas[*v as usize];
            // What the caller can get at and no backward reads is not kept:
            // eager does not check it, and it may be written to.
            let reached = outside.contains(&meta.storage);
            let weight = if reached && !reads.contains(v) { NEVER } else { meta.bytes.max(1) };
            edges.push((into(*v), outof(*v), weight));
            let made = match meta.kind {
                Kind::Forward(at) if !reached && self.runs_again(at as usize, &outside) => Some(at as usize),
                _ => None,
            };
            match made {
                Some(at) => {
                    let mut taken = Vec::new();
                    self.reads(&self.forward[at], &mut taken);
                    edges.extend(taken.iter().map(|u| (outof(*u), into(*v), NEVER)));
                }
                None => edges.push((0, into(*v), NEVER)),
            }
            if reads.contains(v) {
                edges.push((outof(*v), 1, NEVER));
            }
        }
        let reached = min_cut(2 + 2 * count, &edges);
        self.saved = forward.iter().copied().filter(|v| reached[into(*v)] && !reached[outof(*v)]).collect();
        (all, bytes(&self.saved))
    }

    /// The two graphs, for what runs them: a dict with the forward
    /// (`forward`: per operation the operator, its arguments, its keyword
    /// arguments, the values it gives, each with its place in what the
    /// operator returns, and whether it returns one tensor; an
    /// argument is ("v", value), ("c", constant), ("l", list) or ("t",
    /// tuple); the inputs are the values 0 to `inputs`), the values the form
    /// gives (`results`) and which need a gradient (`needs`), what the
    /// forward keeps (`saved`), of those what is checked for writes at the
    /// backward with eager's node and result number (`checked`), whether the
    /// forward can be run again (`draws`, `writes`), and per backward
    /// (`backs`) the results that come with a gradient, the values of those
    /// gradients, its operations (those of the forward it runs again first)
    /// and what it gives: (which contribution, value) in eager's order.
    pub fn describe<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let none = HashMap::new();
        let ops = |ops: &mut dyn Iterator<Item = &Op>| -> PyResult<Bound<'py, PyList>> {
            let described: PyResult<Vec<Bound<'py, PyAny>>> = ops.map(|op| Self::described_op(py, op, &none)).collect();
            PyList::new(py, described?)
        };
        let made = PyDict::new(py);
        made.set_item("inputs", self.inputs)?;
        made.set_item("forward", ops(&mut self.forward.iter())?)?;
        made.set_item("results", self.results.clone())?;
        made.set_item("needs", self.needs.clone())?;
        made.set_item("saved", self.saved.clone())?;
        // What is kept in memory the caller can get at (an input, a result,
        // what shares their memory) is checked for writes, with the node
        // that gave it.
        let outside = self.outside();
        let checked: Vec<(Value, Option<String>, u32)> = self
            .saved
            .iter()
            .filter(|v| outside.contains(&self.metas[**v as usize].storage))
            .map(|v| {
                let producer = &self.metas[*v as usize].producer;
                (*v, producer.as_ref().map(|(name, _)| name.clone()), producer.as_ref().map_or(0, |(_, output)| *output))
            })
            .collect();
        made.set_item("checked", checked)?;
        made.set_item("draws", self.forward.iter().any(|op| op.random))?;
        let outside = |op: &Op| op.writes.iter().any(|v| matches!(self.metas[*v as usize].kind, Kind::Input(_)));
        made.set_item("writes", self.forward.iter().any(outside))?;
        let backs = PyList::empty(py);
        for back in &self.backs {
            backs.append(self.described_part(py, &self.part(&back.pattern, false, None)?)?)?;
        }
        made.set_item("backs", backs)?;
        Ok(made)
    }

    /// How many operations the forward has, and each backward.
    pub fn counts(&self) -> (usize, Vec<usize>) {
        (self.forward.len(), self.backs.iter().map(|back| back.ops.len()).collect())
    }
}
