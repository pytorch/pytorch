//! Graphs: the tensor work of a specialized block as a unit of its own.
//!
//! A graph is a run of the instructions of one specialized block: kernel
//! call sites, its nodes, with what sits between them (reads of globals and
//! attributes, constants, arithmetic on sizes). The graph pass finds the runs
//! of a block once the other passes are done with it and describes each as a
//! `Path`. The VM runs a graph as it runs the block, instruction by
//! instruction and each node through its site, on eager's autograd tape; the
//! path is what `torch.vm.regions` reports.

use crate::capi::{self, SITE_RANDOM, SITE_SCALAR, SITE_WRITES};
use crate::ir::*;
use crate::pyffi::{self, Obj, Shared};
use crate::spec::{CalleeFact, CfgPass, InstFacts, Kernels, PassCx, Unit};
use pyo3::ffi;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// A kernel call site: an operation of the graph.
    Node,
    /// Makes a value from its operands alone: a constant, a size, arithmetic
    /// on numbers, a check.
    Pure,
    /// Reads a global, a cell, or an attribute or item of an object, which
    /// entered no frame and ran no kernel when it was profiled.
    Read,
    /// Keeps values alive, or binds a variable of the frame.
    Residual,
}

pub struct Path {
    /// Where it is entered: the block, and the generic instruction its first
    /// node stands for.
    pub root: (usize, usize),
    /// The operators of its nodes, as the dispatcher names them.
    pub operators: Vec<String>,
}

/// Instructions of a block that are in no graph: the generic instructions
/// lo..hi, and why.
#[derive(Clone)]
pub struct Left {
    pub lo: u32,
    pub hi: u32,
    pub reason: String,
}

/// A graph of one version of a block.
pub struct GraphSite {
    /// The function, the block, and the generic instructions from the first
    /// node to the last.
    pub name: String,
    /// One path: that of the version it was built from.
    pub variants: Shared<Vec<Arc<Path>>>,
    /// The version of the block.
    pub version: AtomicU32,
}

impl GraphSite {
    pub fn install(&self, path: Path, version: u32) {
        self.variants.borrow_mut().push(Arc::new(path));
        self.version.store(version, Ordering::Relaxed);
    }

    pub fn path(&self) -> Option<Arc<Path>> {
        self.variants.borrow().first().cloned()
    }
}

/// An exact int, float or bool.
pub unsafe fn is_number(o: Obj) -> bool {
    !capi::is_unboxed(o)
        && (ffi::PyLong_CheckExact(o) != 0 || ffi::PyFloat_CheckExact(o) != 0 || ffi::PyBool_Check(o) != 0)
}

// ----------------------------------------------------------------------
// The pass
// ----------------------------------------------------------------------

/// The graph pass: finds the graphs of a block, once the other passes are
/// done with it.
pub struct Graphs {
    /// A run with fewer nodes than this is no graph.
    pub min_ops: usize,
}

impl Default for Graphs {
    fn default() -> Graphs {
        Graphs { min_ops: 4 }
    }
}

impl CfgPass for Graphs {
    fn name(&self) -> String {
        "graphs".to_string()
    }

    fn finishes(&self) -> bool {
        true
    }

    fn run(&self, unit: &mut Unit, cx: &PassCx) -> PyResult<bool> {
        Ok(self.build(unit, cx))
    }
}

/// What an instruction is to a graph: a part of one, or what ends it, with
/// the reason.
enum Role {
    Part(Kind),
    Break(String),
}

/// What the run so far knows of a value one of its instructions defines.
#[derive(Clone, Copy)]
struct Def {
    node: bool,
    /// It depends on the result of a node.
    late: bool,
    /// An integer the graph can compute itself.
    graph: bool,
}

/// The binary operators a graph computes on integers: + - * // %.
const INT_BINARY: [u8; 5] = [0, 10, 5, 2, 6];

impl Graphs {
    fn build(&self, unit: &mut Unit, cx: &PassCx) -> bool {
        let n = unit.insts.len();
        let roles: Vec<Role> = (0..n).map(|i| role(unit, cx, i)).collect();
        // The runs, each from its first node to its last.
        let mut runs: Vec<(usize, usize)> = Vec::new();
        let mut at = 0;
        while at < n {
            if matches!(roles[at], Role::Break(_)) {
                at += 1;
                continue;
            }
            let end = stretch(unit, cx, &roles, at);
            let is_node = |i: &usize| matches!(roles[*i], Role::Part(Kind::Node));
            if let (Some(first), Some(last)) = ((at..end).find(is_node), (at..end).rev().find(is_node)) {
                runs.push((first, last + 1));
            }
            at = end;
        }
        let origins: Vec<u32> = unit.insts.iter().map(|u| u.origin).collect();
        // What is reported as left out: the instructions that end a run
        // where the block has tensor work on both sides, and the runs that
        // are too short to be graphs.
        let works = |i: &usize| {
            let facts = cx.facts.get(origins[*i] as usize);
            matches!(unit.insts[*i].inst.op, Op::Dispatch(_))
                || facts.is_some_and(|f| matches!(f.kernels, Kernels::One | Kernels::Many))
        };
        let mut left: Vec<Left> = Vec::new();
        if let (Some(first), Some(last)) = ((0..n).find(works), (0..n).rev().find(works)) {
            for i in first..=last {
                if let (Role::Break(reason), true) = (&roles[i], origins[i] != NONE) {
                    left.push(Left { lo: origins[i], hi: origins[i] + 1, reason: reason.clone() });
                }
            }
        }
        let mut found: Vec<Path> = Vec::new();
        for (first, end) in runs {
            let nodes = (first..end).filter(|i| matches!(roles[*i], Role::Part(Kind::Node))).count();
            if nodes < self.min_ops {
                let reason = format!("{} operations, fewer than min_ops ({})", nodes, self.min_ops);
                left.push(Left { lo: origins[first], hi: origins[end - 1] + 1, reason });
                continue;
            }
            found.push(path(unit, first, end));
        }
        unit.left = left;
        for path in found {
            let name = format!("{} (block {})[{}]", cx.func().qualname, unit.block, path.root.1);
            let site = Arc::new(GraphSite {
                name,
                variants: Shared::new(Vec::new()),
                version: AtomicU32::new(0),
            });
            unit.graphs.push((site, path));
        }
        // The instructions stay as they are: the VM runs a graph as it runs
        // the block.
        false
    }
}

/// An exact number among the constants of the function.
fn const_number(cx: &PassCx, unit: &Unit, v: V, int: bool) -> bool {
    let Some(Op::Const(c)) = unit.def_op(v) else {
        return false;
    };
    let obj = cx.func().consts[*c as usize].as_ptr();
    unsafe { if int { ffi::PyLong_CheckExact(obj) != 0 } else { is_number(obj) } }
}

/// Whether a value was an exact number whenever it was looked at: a constant
/// that is one, a size, or what the profile saw.
fn number(cx: &PassCx, unit: &Unit, v: V) -> bool {
    let sized = matches!(unit.def_op(v), Some(Op::TensorSize { .. } | Op::TensorDim(_)));
    sized || const_number(cx, unit, v, false) || cx.obs.get(v as usize).is_some_and(|o| o.is_number())
}

fn int_like(cx: &PassCx, unit: &Unit, v: V) -> bool {
    let sized = matches!(unit.def_op(v), Some(Op::TensorSize { .. } | Op::TensorDim(_)));
    let seen = matches!(cx.obs.get(v as usize), Some(crate::spec::ValueObs::Int { .. }));
    sized || const_number(cx, unit, v, true) || seen
}

fn facts<'a>(cx: &'a PassCx, unit: &Unit, i: usize) -> Option<&'a InstFacts> {
    cx.facts.get(unit.insts[i].origin as usize)
}

fn role(unit: &Unit, cx: &PassCx, i: usize) -> Role {
    let quiet = facts(cx, unit, i).is_some_and(|f| f.quiet());
    let wrote = facts(cx, unit, i).is_some_and(|f| f.wrote);
    let numbers = |vs: &[V]| vs.iter().all(|v| number(cx, unit, *v));
    let kind = match &unit.insts[i].inst.op {
        Op::Dispatch(d) if d.slot.traits() == 0 && d.slot.len() > 0 && !wrote => Some(Kind::Node),
        Op::Const(_)
        | Op::TensorSize { .. }
        | Op::TensorDim(_)
        | Op::TupleGet(..)
        | Op::IsNone(..)
        | Op::Is(..)
        | Op::IsFunction { .. }
        | Op::BuildTuple(_)
        | Op::CheckBound(..)
        | Op::Guard(_) => Some(Kind::Pure),
        Op::Binary(_, a, b) | Op::Compare(_, a, b) if numbers(&[*a, *b]) => Some(Kind::Pure),
        Op::Unary(_, a) | Op::ToBool(a) if numbers(&[*a]) => Some(Kind::Pure),
        Op::LoadGlobal(_) | Op::LoadDeref(_) | Op::CellGet(..) => Some(Kind::Read),
        Op::LoadAttr(..) | Op::GetItem(..) | Op::Contains(..) | Op::GetLen(_) | Op::Unpack(..) if quiet => {
            Some(Kind::Read)
        }
        Op::Hold(_) | Op::StoreLocal(..) => Some(Kind::Residual),
        _ => None,
    };
    match kind {
        Some(kind) => Role::Part(kind),
        None => Role::Break(why(unit, cx, i)),
    }
}

/// Why an instruction is in no graph.
fn why(unit: &Unit, cx: &PassCx, i: usize) -> String {
    let u = &unit.insts[i];
    let found = facts(cx, unit, i);
    let no_site = || match cx.reasons.get(u.origin as usize).and_then(|r| r.as_ref()) {
        Some(reason) => format!("runs kernels and is no call site: {}", reason),
        None => "runs kernels and is no call site".to_string(),
    };
    let kernels = found.is_some_and(|f| matches!(f.kernels, Kernels::One | Kernels::Many));
    match &u.inst.op {
        Op::Dispatch(d) => {
            let operator = unsafe { d.slot.operator() }.unwrap_or_else(|| "the kernel".to_string());
            let traits = d.slot.traits();
            if traits & SITE_WRITES != 0 || found.is_some_and(|f| f.wrote) {
                format!("{} writes to an argument", operator)
            } else if traits & SITE_RANDOM != 0 {
                format!("{} draws random numbers", operator)
            } else if traits & SITE_SCALAR != 0 {
                format!("{} gives a number read from a tensor", operator)
            } else {
                format!("{} has no call site", operator)
            }
        }
        Op::Call { .. } | Op::CallMethod { .. } | Op::CallEx { .. } => match found.map(|f| &f.callee) {
            Some(CalleeFact::One(code, _)) => {
                let name = unsafe {
                    let name = ffi::PyObject_GetAttrString(code.as_ptr(), c"co_qualname".as_ptr());
                    if name.is_null() {
                        ffi::PyErr_Clear();
                        "a function".to_string()
                    } else {
                        let text = pyffi::str_to_string(name);
                        ffi::Py_DECREF(name);
                        text
                    }
                };
                format!("calls {}", name)
            }
            Some(CalleeFact::Many) => "calls more than one function".to_string(),
            _ if kernels => no_site(),
            _ if found.is_some_and(|f| f.frames) => "calls Python code".to_string(),
            _ if found.is_some_and(|f| f.runs > 0) => "a call that runs no kernel".to_string(),
            _ => "a call that was not profiled".to_string(),
        },
        Op::StoreAttr { .. }
        | Op::SetItem { .. }
        | Op::DelItem(..)
        | Op::StoreGlobal(..)
        | Op::StoreDeref(..)
        | Op::CellSet(..) => {
            if found.is_some_and(|f| f.wrote) {
                "writes to a tensor".to_string()
            } else if kernels {
                no_site()
            } else {
                "stores into an object".to_string()
            }
        }
        op => {
            let text = cx.func().op_repr(op);
            let name = text.split_whitespace().next().unwrap_or("it");
            if kernels {
                no_site()
            } else if found.is_some_and(|f| f.frames) {
                format!("{} enters Python code", name)
            } else if found.is_some_and(|f| f.raised) {
                format!("{} raised", name)
            } else {
                format!("{} is no part of a graph", name)
            }
        }
    }
}

/// What an instruction of a run reads, for where it stands in the run: the
/// operands of a site, not what its fallback would take.
fn reads(op: &Op, mut f: impl FnMut(V)) {
    match op {
        Op::Dispatch(d) => d.operands.iter().for_each(|v| f(*v)),
        Op::Hold(_) | Op::StoreLocal(..) => {}
        op => op.for_each_use(f),
    }
}

/// Whether the graph can compute what a late instruction gives: a size of a
/// tensor of its own, or arithmetic on such sizes and other integers.
fn computes(op: &Op, defs: &HashMap<V, Def>, cx: &PassCx, unit: &Unit) -> bool {
    let fits = |v: &V| match defs.get(v) {
        Some(def) if def.late => def.graph,
        _ => int_like(cx, unit, *v),
    };
    match op {
        Op::TensorSize { t, .. } | Op::TensorDim(t) => defs.get(t).is_some_and(|def| def.node),
        Op::Binary(k, a, b) => INT_BINARY.contains(k) && fits(a) && fits(b),
        Op::Unary(UnaryKind::Neg, a) => fits(a),
        Op::BuildTuple(vs) => vs.iter().all(fits),
        _ => false,
    }
}

/// The definitions of a run as far as it has got, with the instruction at
/// `i` added. False if the instruction cannot be in the run: a node that
/// takes something that was computed from a node and that the graph cannot
/// compute itself, or a guard on such a thing.
fn define(unit: &Unit, cx: &PassCx, roles: &[Role], defs: &mut HashMap<V, Def>, i: usize) -> bool {
    let inst = &unit.insts[i].inst;
    let node = matches!(roles[i], Role::Part(Kind::Node));
    let mut late = node;
    let mut fits = true;
    reads(&inst.op, |v| {
        if let Some(def) = defs.get(&v) {
            late |= def.late;
            fits &= !def.late || def.node || def.graph;
        }
    });
    let guarded = match &inst.op {
        Op::Guard(g) => defs.get(&g.cond.value()).is_some_and(|def| def.late && !def.node),
        _ => false,
    };
    if (node && !fits) || guarded {
        return false;
    }
    if inst.dst != NONE {
        let graph = late && !node && computes(&inst.op, defs, cx, unit);
        defs.insert(inst.dst, Def { node, late, graph });
    }
    true
}

/// How far a run that starts at `at` goes: up to the first instruction that
/// ends it.
fn stretch(unit: &Unit, cx: &PassCx, roles: &[Role], at: usize) -> usize {
    let mut defs: HashMap<V, Def> = HashMap::new();
    let mut i = at;
    while i < unit.insts.len() {
        if matches!(roles[i], Role::Break(_)) || !define(unit, cx, roles, &mut defs, i) {
            break;
        }
        i += 1;
    }
    // An instruction that cannot start a run either would never be passed.
    i.max(at + 1)
}

/// The path of the instructions first..end of a unit, which start and end
/// with a node.
fn path(unit: &Unit, first: usize, end: usize) -> Path {
    let mut operators = Vec::new();
    for u in &unit.insts[first..end] {
        if let Op::Dispatch(d) = &u.inst.op {
            operators.push(unsafe { d.slot.operator() }.unwrap_or_else(|| "?".to_string()));
        }
    }
    Path {
        root: (unit.block, unit.insts[first].origin as usize),
        operators,
    }
}
