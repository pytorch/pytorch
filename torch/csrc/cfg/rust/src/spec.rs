//! Specialization of basic blocks: the unit passes work on, the pass
//! pipeline, the built-in passes, and the checks that turn a transformed unit
//! into an executable version of its block.
//!
//! A region (today: one basic block) is specialized after it ran in the
//! profiling tier. Passes rewrite a copy of its generic instructions, using
//! what the profile observed, and may attach guards. A guard on a value that
//! is already there when the block is entered becomes an entry guard of the
//! version; a guard on a value the block computes becomes a `Guard`
//! instruction that resumes the generic code at a given instruction. Since a
//! failed guard only ever sends this block back to the profiling tier, only
//! this block is recompiled.

use crate::analysis::BitSet;
use crate::capi::{self, SiteSlot, TCFG_METHOD};
use crate::ir::*;
use crate::pyffi::{self, PyObj};
use crate::vm::Compiled;
use pyo3::prelude::*;
use std::sync::Arc;

/// What the profiling tier saw a value be.
#[derive(Clone, Default, Debug)]
pub enum ValueObs {
    #[default]
    Unseen,
    /// Always a tensor. `dims` is the observed [min, max] of each dimension,
    /// or empty if the rank varied.
    Tensor { min_rank: u32, dims: Vec<(i64, i64)>, last: Vec<i64> },
    Bool { seen_true: u32, seen_false: u32, last: bool },
    Int { lo: i64, hi: i64 },
    Other,
}

impl ValueObs {
    pub unsafe fn observe(&mut self, obj: pyffi::Obj) {
        use pyo3::ffi;
        if obj.is_null() {
            return;
        }
        // An unboxed tensor is not a Python object: only the tensor check at
        // the end may look at it.
        let python = !capi::is_unboxed(obj);
        let new = if python && (obj == ffi::Py_True() || obj == ffi::Py_False()) {
            let t = obj == ffi::Py_True();
            ValueObs::Bool { seen_true: t as u32, seen_false: !t as u32, last: t }
        } else if python && ffi::PyLong_CheckExact(obj) != 0 {
            let mut overflow = 0;
            let x = ffi::PyLong_AsLongLongAndOverflow(obj, &mut overflow);
            if overflow != 0 {
                ValueObs::Other
            } else {
                ValueObs::Int { lo: x, hi: x }
            }
        } else {
            let mut info = capi::TcfgTensorInfo::empty();
            match capi::api() {
                Some(api) if (api.tensor_info)(obj, &mut info) != 0 => {
                    let sizes: Vec<i64> =
                        (0..info.ndim as isize).map(|i| *info.sizes.offset(i)).collect();
                    ValueObs::Tensor {
                        min_rank: info.ndim as u32,
                        dims: sizes.iter().map(|s| (*s, *s)).collect(),
                        last: sizes,
                    }
                }
                _ => ValueObs::Other,
            }
        };
        let merged = match (&*self, new) {
            (ValueObs::Unseen, new) => new,
            (
                ValueObs::Tensor { min_rank, dims, .. },
                ValueObs::Tensor { min_rank: r2, dims: d2, last },
            ) => {
                let dims = if dims.len() == d2.len() && *min_rank == r2 {
                    dims.iter()
                        .zip(&d2)
                        .map(|((lo, hi), (x, _))| (*lo.min(x), *hi.max(x)))
                        .collect()
                } else {
                    Vec::new()
                };
                ValueObs::Tensor { min_rank: (*min_rank).min(r2), dims, last }
            }
            (
                ValueObs::Bool { seen_true, seen_false, .. },
                ValueObs::Bool { seen_true: t, seen_false: f, last },
            ) => ValueObs::Bool { seen_true: seen_true + t, seen_false: seen_false + f, last },
            (ValueObs::Int { lo, hi }, ValueObs::Int { lo: x, .. }) => {
                ValueObs::Int { lo: (*lo).min(x), hi: (*hi).max(x) }
            }
            _ => ValueObs::Other,
        };
        *self = merged;
    }

    pub fn tensor_rank(&self) -> Option<u32> {
        match self {
            ValueObs::Tensor { min_rank, .. } => Some(*min_rank),
            _ => None,
        }
    }
}

pub struct UInst {
    pub inst: Inst,
    /// Index of the generic instruction this one stands for, or NONE.
    pub origin: u32,
    /// The result may be dropped when unused even though the generic
    /// operation is not pure in general (its consumer took over its effect,
    /// or a guard makes it pure).
    pub removable: bool,
}

/// The code of a block while passes transform it.
pub struct Unit {
    pub block: usize,
    pub insts: Vec<UInst>,
    pub term: Term,
    pub entry_guards: Vec<GuardCond>,
    /// Whether passes may add guards based on what was observed. Turned off
    /// for a block that keeps failing its guards, so that it settles on code
    /// that holds for every input.
    pub speculate: bool,
}

impl Unit {
    pub fn from_generic(func: &Function, block: usize, speculate: bool) -> Unit {
        let b = &func.blocks[block];
        Unit {
            block,
            insts: b
                .insts
                .iter()
                .enumerate()
                .map(|(i, inst)| UInst {
                    inst: Inst { kills: Vec::new(), ..inst.clone() },
                    origin: i as u32,
                    removable: false,
                })
                .collect(),
            term: b.term.clone(),
            entry_guards: Vec::new(),
            speculate,
        }
    }

    /// Position of the instruction defining `v` in this unit.
    pub fn def_pos(&self, v: V) -> Option<usize> {
        self.insts.iter().position(|u| u.inst.dst == v)
    }

    pub fn def_op(&self, v: V) -> Option<&Op> {
        self.def_pos(v).map(|p| &self.insts[p].inst.op)
    }

    /// Makes the version depend on `cond`. A value that exists when the block
    /// is entered is checked there; otherwise the check goes right before
    /// position `before`, resuming the generic code at that instruction.
    /// Returns None if the guard cannot be placed, and otherwise whether the
    /// unit changed (false: the version already depended on `cond`).
    pub fn add_guard(&mut self, cx: &PassCx, cond: GuardCond, before: usize) -> Option<bool> {
        let v = cond.value();
        if self.def_pos(v).is_none() {
            if !cx.cf.analysis.available_at_entry(self.block, v) {
                return None;
            }
            return Some(self.add_entry_guard(cond));
        }
        if self.def_pos(v).unwrap() >= before {
            return None;
        }
        let resume = match self.insts.get(before) {
            Some(u) if u.origin != NONE => u.origin,
            _ => return None,
        };
        if let Some(Op::Guard(g)) = before.checked_sub(1).map(|p| &self.insts[p].inst.op) {
            if g.cond == cond {
                return Some(false);
            }
        }
        let bc = self.insts[before].inst.bc;
        self.insts.insert(
            before,
            UInst {
                inst: Inst {
                    dst: NONE,
                    op: Op::Guard(Box::new(GuardOp { cond, resume, live: Vec::new() })),
                    bc,
                    kills: Vec::new(),
                },
                origin: NONE,
                removable: false,
            },
        );
        Some(true)
    }

    /// Adds an entry guard, merging it with one on the same quantity. Returns
    /// whether the entry guards changed.
    fn add_entry_guard(&mut self, cond: GuardCond) -> bool {
        for existing in self.entry_guards.iter_mut() {
            let before = existing.clone();
            match (&mut *existing, &cond) {
                (
                    GuardCond::TensorRank { v, min_rank },
                    GuardCond::TensorRank { v: v2, min_rank: r2 },
                ) if v == v2 => *min_rank = (*min_rank).max(*r2),
                (
                    GuardCond::DimRange { v, dim, lo, hi },
                    GuardCond::DimRange { v: v2, dim: d2, lo: lo2, hi: hi2 },
                ) if v == v2 && dim == d2 => {
                    *lo = (*lo).max(*lo2);
                    *hi = (*hi).min(*hi2);
                }
                _ => continue,
            }
            return *existing != before;
        }
        self.entry_guards.push(cond);
        true
    }

    pub fn dump(&self, func: &Function) -> String {
        let mut out = String::new();
        for g in &self.entry_guards {
            out.push_str(&format!("    entry guard {}\n", func.guard_repr(g)));
        }
        let insts: Vec<Inst> = self.insts.iter().map(|u| u.inst.clone()).collect();
        func.dump_code(&mut out, &insts, &self.term);
        out
    }
}

/// What a pass can look at besides the unit: the whole generic graph with its
/// analyses, and the profile.
pub struct PassCx<'a> {
    pub cf: &'a Compiled,
    /// Per generic instruction of the block: its dispatch sites, if the
    /// profiling tier found the instruction to be a single dispatcher call.
    pub slots: &'a [Option<Arc<SiteSlot>>],
    /// Per SSA value of the function.
    pub obs: &'a [ValueObs],
}

impl PassCx<'_> {
    pub fn func(&self) -> &Function {
        &self.cf.func
    }

    fn const_int(&self, unit: &Unit, v: V) -> Option<i64> {
        let Some(Op::Const(c)) = unit.def_op(v) else {
            return None;
        };
        let obj = self.func().consts[*c as usize].as_ptr();
        unsafe {
            use pyo3::ffi;
            if ffi::PyLong_CheckExact(obj) == 0 {
                return None;
            }
            let mut overflow = 0;
            let x = ffi::PyLong_AsLongLongAndOverflow(obj, &mut overflow);
            (overflow == 0).then_some(x)
        }
    }

    fn name_is(&self, name: u32, expected: &str) -> bool {
        unsafe { pyffi::str_to_string(self.func().consts[name as usize].as_ptr()) == expected }
    }
}

/// A CFG compiler pass. Returns whether it changed the unit; the pipeline
/// keeps running its passes until none of them does.
pub trait CfgPass: Send + Sync {
    fn name(&self) -> String;
    fn run(&self, unit: &mut Unit, cx: &PassCx) -> PyResult<bool>;
}

impl Pipeline {
    pub fn names(&self) -> Vec<String> {
        self.passes.iter().map(|p| p.name()).collect()
    }
}

pub fn builtin_pass(name: &str) -> Option<Box<dyn CfgPass>> {
    match name {
        "resolve_dispatch" => Some(Box::new(ResolveDispatch)),
        "method_calls" => Some(Box::new(MethodCalls)),
        "tensor_accessors" => Some(Box::new(TensorAccessors)),
        "narrow_dims" => Some(Box::new(NarrowDims)),
        "fold" => Some(Box::new(Fold)),
        "dce" => Some(Box::new(Dce)),
        _ => None,
    }
}

pub const BUILTIN_PASSES: [&str; 6] =
    ["resolve_dispatch", "method_calls", "tensor_accessors", "narrow_dims", "fold", "dce"];

/// Replaces operations the profile found to be a single dispatcher call with
/// a direct call of the post-dispatch kernel.
struct ResolveDispatch;

impl CfgPass for ResolveDispatch {
    fn name(&self) -> String {
        "resolve_dispatch".to_string()
    }

    fn run(&self, unit: &mut Unit, cx: &PassCx) -> PyResult<bool> {
        let mut changed = false;
        for pos in 0..unit.insts.len() {
            let origin = unit.insts[pos].origin;
            if origin == NONE || matches!(unit.insts[pos].inst.op, Op::Dispatch(_)) {
                continue;
            }
            let Some(Some(slot)) = cx.slots.get(origin as usize) else {
                continue;
            };
            if slot.len() == 0 {
                continue;
            }
            let op = unit.insts[pos].inst.op.clone();
            let Some(operands) = cx.cf.site_operands(slot.kind, &op) else {
                continue;
            };
            let mut fallback = op.clone();
            if slot.kind == TCFG_METHOD {
                // The kernel call does not need the bound method object. If
                // nothing else does either, drop its load and let the
                // fallback look the method up again.
                if let Op::Call { callee, args, kwnames } = &op {
                    if let (Some(lp), true) = (
                        unit.def_pos(*callee),
                        cx.cf.analysis.use_count[*callee as usize] == 1,
                    ) {
                        if let Op::LoadAttr(recv, name) = unit.insts[lp].inst.op.clone() {
                            fallback = Op::CallMethod {
                                recv,
                                name,
                                args: args.clone(),
                                kwnames: *kwnames,
                            };
                            unit.insts[lp].removable = true;
                        }
                    }
                }
            }
            unit.insts[pos].inst.op =
                Op::Dispatch(Box::new(DispatchOp { slot: slot.clone(), operands, fallback }));
            changed = true;
        }
        Ok(changed)
    }
}

/// Fuses `m = x.name` with the call `m(...)` it feeds into a single method
/// call, which does not need the bound method object. Only done when nothing
/// but loads of constants and variables sits between the two, so that the
/// lookup moving down to the call cannot be observed.
struct MethodCalls;

impl CfgPass for MethodCalls {
    fn name(&self) -> String {
        "method_calls".to_string()
    }

    fn run(&self, unit: &mut Unit, cx: &PassCx) -> PyResult<bool> {
        let mut changed = false;
        for pos in 0..unit.insts.len() {
            let Op::Call { callee, args, kwnames } = &unit.insts[pos].inst.op else {
                continue;
            };
            let (callee, args, kwnames) = (*callee, args.clone(), *kwnames);
            let Some(lp) = unit.def_pos(callee) else {
                continue;
            };
            let Op::LoadAttr(recv, name) = unit.insts[lp].inst.op else {
                continue;
            };
            let inert = unit.insts[lp + 1..pos].iter().all(|u| {
                matches!(u.inst.op, Op::Const(_) | Op::LoadGlobal(_) | Op::LoadDeref(_))
            });
            if cx.cf.analysis.use_count[callee as usize] != 1 || !inert || args.contains(&callee) {
                continue;
            }
            unit.insts[pos].inst.op = Op::CallMethod { recv, name, args, kwnames };
            unit.insts[lp].removable = true;
            changed = true;
        }
        Ok(changed)
    }
}

/// Turns `x.shape[i]`, `x.size(i)`, `x.dim()` and `x.ndim` on values that
/// have only ever been tensors into direct reads of the tensor, guarded on the
/// value being a tensor of sufficient rank. This is what exposes dimensions
/// to the rest of the pipeline as plain SSA integers.
struct TensorAccessors;

enum Accessor {
    Size(i64),
    Dim,
}

impl TensorAccessors {
    /// Matches the instruction at `pos`; returns the tensor, what is read,
    /// and the position of the attribute load feeding it (if separate).
    fn match_at(unit: &Unit, cx: &PassCx, pos: usize) -> Option<(V, Accessor, Option<usize>)> {
        let attr_load = |v: V, attr: &str| -> Option<(V, usize)> {
            let lp = unit.def_pos(v)?;
            match &unit.insts[lp].inst.op {
                Op::LoadAttr(x, name)
                    if cx.name_is(*name, attr) && cx.cf.analysis.use_count[v as usize] == 1 =>
                {
                    Some((*x, lp))
                }
                _ => None,
            }
        };
        match &unit.insts[pos].inst.op {
            Op::GetItem(s, k) => {
                let (x, lp) = attr_load(*s, "shape")?;
                Some((x, Accessor::Size(cx.const_int(unit, *k)?), Some(lp)))
            }
            Op::Call { callee, args, kwnames } if *kwnames == NONE => {
                if args.len() == 1 {
                    let (x, lp) = attr_load(*callee, "size")?;
                    Some((x, Accessor::Size(cx.const_int(unit, args[0])?), Some(lp)))
                } else if args.is_empty() {
                    let (x, lp) = attr_load(*callee, "dim")?;
                    Some((x, Accessor::Dim, Some(lp)))
                } else {
                    None
                }
            }
            Op::LoadAttr(x, name) if cx.name_is(*name, "ndim") => Some((*x, Accessor::Dim, None)),
            _ => None,
        }
    }
}

impl CfgPass for TensorAccessors {
    fn name(&self) -> String {
        "tensor_accessors".to_string()
    }

    fn run(&self, unit: &mut Unit, cx: &PassCx) -> PyResult<bool> {
        if !unit.speculate {
            return Ok(false);
        }
        let mut changed = false;
        let mut pos = 0;
        while pos < unit.insts.len() {
            let Some((x, accessor, load_pos)) = Self::match_at(unit, cx, pos) else {
                pos += 1;
                continue;
            };
            let Some(rank) = cx.obs[x as usize].tensor_rank() else {
                pos += 1;
                continue;
            };
            let min_rank = match accessor {
                Accessor::Size(d) => (if d >= 0 { d + 1 } else { -d }) as u32,
                Accessor::Dim => 0,
            };
            // Between the attribute load and its use only constants may be
            // loaded: the read moves from the former to the latter.
            let first = load_pos.unwrap_or(pos);
            let only_consts = unit.insts[first + 1..pos]
                .iter()
                .all(|u| matches!(u.inst.op, Op::Const(_)));
            if rank < min_rank || !only_consts {
                pos += 1;
                continue;
            }
            if unit.add_guard(cx, GuardCond::TensorRank { v: x, min_rank }, first).is_none() {
                pos += 1;
                continue;
            }
            // A guard instruction may have been inserted before `first`.
            let (first, pos_now) = if matches!(unit.insts[first].inst.op, Op::Guard(_)) {
                (first + 1, pos + 1)
            } else {
                (first, pos)
            };
            if load_pos.is_some() {
                unit.insts[first].removable = true;
            }
            unit.insts[pos_now].inst.op = match accessor {
                Accessor::Size(dim) => Op::TensorSize { t: x, dim },
                Accessor::Dim => Op::TensorDim(x),
            };
            changed = true;
            pos = pos_now + 1;
        }
        Ok(changed)
    }
}

/// Narrows dimensions that control flow depends on. A comparison of a tensor
/// dimension with a constant that the profile last saw go one way is folded
/// to that outcome, under a guard that pins the dimension to the range of
/// sizes for which the outcome holds. Everything is dynamic until a pass like
/// this one decides a range is worth a guard.
struct NarrowDims;

impl NarrowDims {
    /// The range of `size` for which `size <cmp> c` evaluates to `outcome`.
    fn range_for(cmp: u8, c: i64, outcome: bool, observed: i64) -> Option<(i64, i64)> {
        const INF: i64 = i64::MAX;
        // < <= == != > >=
        let (cmp, outcome) = match (cmp, outcome) {
            (3, o) => (2, !o),
            other => other,
        };
        let range = match (cmp, outcome) {
            (0, true) => (0, c.checked_sub(1)?),
            (0, false) => (c, INF),
            (1, true) => (0, c),
            (1, false) => (c.checked_add(1)?, INF),
            (2, true) => (c, c),
            (2, false) => {
                if observed > c {
                    (c.checked_add(1)?, INF)
                } else {
                    (0, c.checked_sub(1)?)
                }
            }
            (4, true) => (c.checked_add(1)?, INF),
            (4, false) => (0, c),
            (5, true) => (c, INF),
            (5, false) => (0, c.checked_sub(1)?),
            _ => return None,
        };
        let range = (range.0.max(0), range.1);
        (range.0 <= range.1 && observed >= range.0 && observed <= range.1).then_some(range)
    }
}

impl CfgPass for NarrowDims {
    fn name(&self) -> String {
        "narrow_dims".to_string()
    }

    fn run(&self, unit: &mut Unit, cx: &PassCx) -> PyResult<bool> {
        if !unit.speculate {
            return Ok(false);
        }
        let mut changed = false;
        let mut pos = 0;
        while pos < unit.insts.len() {
            let (cmp, a, b) = match &unit.insts[pos].inst.op {
                Op::Compare(cmp, a, b) => (*cmp, *a, *b),
                _ => {
                    pos += 1;
                    continue;
                }
            };
            let dst = unit.insts[pos].inst.dst;
            // size <cmp> const, or const <cmp> size mirrored.
            let mirrored = [4u8, 5, 2, 3, 0, 1];
            let found = match (unit.def_op(a), unit.def_op(b)) {
                (Some(Op::TensorSize { t, dim }), _) => {
                    cx.const_int(unit, b).map(|c| (*t, *dim, cmp, c))
                }
                (_, Some(Op::TensorSize { t, dim })) => {
                    cx.const_int(unit, a).map(|c| (*t, *dim, mirrored[cmp as usize], c))
                }
                _ => None,
            };
            let outcome = match &cx.obs[dst as usize] {
                ValueObs::Bool { last, .. } => Some(*last),
                _ => None,
            };
            let observed = found.and_then(|(t, dim, _, _)| match &cx.obs[t as usize] {
                ValueObs::Tensor { last, .. } => {
                    let d = if dim < 0 { dim + last.len() as i64 } else { dim };
                    last.get(d as usize).copied()
                }
                _ => None,
            });
            let (Some((t, dim, cmp, c)), Some(outcome), Some(observed)) = (found, outcome, observed)
            else {
                pos += 1;
                continue;
            };
            let Some((lo, hi)) = Self::range_for(cmp, c, outcome, observed) else {
                pos += 1;
                continue;
            };
            let before = unit.insts.len();
            if unit.add_guard(cx, GuardCond::DimRange { v: t, dim, lo, hi }, pos).is_none() {
                pos += 1;
                continue;
            }
            let pos_now = pos + (unit.insts.len() - before);
            let func = cx.func();
            unit.insts[pos_now].inst.op =
                Op::Const(if outcome { func.const_true } else { func.const_false });
            changed = true;
            pos = pos_now + 1;
        }
        Ok(changed)
    }
}

/// Folds truthiness tests and branches on constants.
struct Fold;

impl CfgPass for Fold {
    fn name(&self) -> String {
        "fold".to_string()
    }

    fn run(&self, unit: &mut Unit, cx: &PassCx) -> PyResult<bool> {
        let func = cx.func();
        let const_bool = |unit: &Unit, v: V| -> Option<bool> {
            match unit.def_op(v) {
                Some(Op::Const(c)) if *c == func.const_true => Some(true),
                Some(Op::Const(c)) if *c == func.const_false => Some(false),
                _ => None,
            }
        };
        let mut changed = false;
        for pos in 0..unit.insts.len() {
            if let Op::ToBool(a) = unit.insts[pos].inst.op {
                if let Some(value) = const_bool(unit, a) {
                    unit.insts[pos].inst.op =
                        Op::Const(if value { func.const_true } else { func.const_false });
                    changed = true;
                }
            }
        }
        if let Term::Br { cond, then_, else_ } = unit.term {
            if let Some(value) = const_bool(unit, cond) {
                unit.term = Term::Jump(if value { then_ } else { else_ });
                changed = true;
            }
        }
        Ok(changed)
    }
}

/// Drops instructions whose result nobody needs: neither the unit, nor the
/// blocks after it, nor the generic code a guard may resume.
struct Dce;

impl CfgPass for Dce {
    fn name(&self) -> String {
        "dce".to_string()
    }

    fn run(&self, unit: &mut Unit, cx: &PassCx) -> PyResult<bool> {
        let func = cx.func();
        let analysis = &cx.cf.analysis;
        let mut needed = analysis.live_out[unit.block].clone();
        unit.term.for_each_use(|v| {
            needed.insert(v);
        });
        let mut keep = vec![true; unit.insts.len()];
        for (pos, u) in unit.insts.iter().enumerate().rev() {
            if let Op::Guard(g) = &u.inst.op {
                needed.insert(g.cond.value());
                needed.union_with(&analysis.live_before(func, unit.block, g.resume as usize));
                continue;
            }
            let dst = u.inst.dst;
            if dst != NONE && !needed.contains(dst) && (u.removable || u.inst.op.is_pure()) {
                keep[pos] = false;
                continue;
            }
            if dst != NONE {
                needed.remove(dst);
            }
            u.inst.op.for_each_use(|v| {
                needed.insert(v);
            });
        }
        let before = unit.insts.len();
        let mut pos = 0;
        unit.insts.retain(|_| {
            pos += 1;
            keep[pos - 1]
        });
        Ok(unit.insts.len() != before)
    }
}

/// A pass written in Python: a callable taking a `Region` and returning
/// whether it changed it.
pub struct PyPass {
    pub callable: PyObj,
    pub name: String,
}

impl CfgPass for PyPass {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn run(&self, unit: &mut Unit, cx: &PassCx) -> PyResult<bool> {
        crate::pyapi::run_python_pass(&self.callable, unit, cx)
    }
}

pub struct Pipeline {
    pub passes: Vec<Box<dyn CfgPass>>,
}

const MAX_ROUNDS: usize = 8;

impl Pipeline {
    /// Runs the passes in order, again and again, until the unit stops
    /// changing. Returns whether anything changed at all.
    pub fn run(&self, unit: &mut Unit, cx: &PassCx) -> PyResult<bool> {
        let mut any = false;
        for _ in 0..MAX_ROUNDS {
            let mut changed = false;
            for pass in &self.passes {
                changed |= pass.run(unit, cx)?;
            }
            any |= changed;
            if !changed {
                break;
            }
        }
        Ok(any)
    }
}

/// Turns a unit into executable code, or reports why the passes left it in a
/// state that cannot run.
pub fn finalize(unit: Unit, cf: &Compiled, id: u32) -> Result<BlockCode, String> {
    let func = &cf.func;
    let analysis = &cf.analysis;
    let b = unit.block;
    let n = func.nvalues as usize;

    let mut available: BitSet = analysis.live_in[b].clone();
    for g in &unit.entry_guards {
        if !available.contains(g.value()) {
            return Err(format!(
                "entry guard on {} which is not available at block entry",
                func.value_name(g.value())
            ));
        }
    }
    let mut insts: Vec<Inst> = Vec::with_capacity(unit.insts.len());
    // Specialized code must run the generic instructions it keeps in their
    // original order, and a guard must resume exactly at the first generic
    // instruction that has not run yet.
    let mut last_origin: Option<u32> = None;
    let mut last_resume: u32 = 0;
    for u in unit.insts {
        let mut inst = u.inst;
        if let Op::Guard(g) = &mut inst.op {
            if (g.resume as usize) > func.blocks[b].insts.len() {
                return Err("guard resumes past the end of the block".to_string());
            }
            if last_origin.is_some_and(|o| o >= g.resume) {
                return Err("guard resumes before an instruction that already ran".to_string());
            }
            last_resume = last_resume.max(g.resume);
            let live = analysis.live_before(func, b, g.resume as usize);
            g.live = live.iter().collect();
        }
        if u.origin != NONE {
            if last_origin.is_some_and(|o| o >= u.origin) {
                return Err("instructions were reordered".to_string());
            }
            if u.origin < last_resume {
                return Err("a guard would skip an instruction when it resumes".to_string());
            }
            last_origin = Some(u.origin);
        }
        let mut missing = None;
        inst.op.for_each_use(|v| {
            if v != UNDEF && (v as usize >= n || !available.contains(v)) {
                missing = Some(v);
            }
        });
        if let Some(v) = missing {
            return Err(format!("{} is used before it is defined", func.value_name(v)));
        }
        if inst.dst != NONE {
            available.insert(inst.dst);
        }
        insts.push(inst);
    }
    let mut missing = None;
    unit.term.for_each_use(|v| {
        if v != UNDEF && !available.contains(v) {
            missing = Some(v);
        }
    });
    for v in analysis.live_out[b].iter() {
        if !available.contains(v) {
            missing = Some(v);
        }
    }
    if let Some(v) = missing {
        return Err(format!("{} is needed after the block but not defined", func.value_name(v)));
    }
    let term_kills = crate::analysis::compute_kills(&mut insts, &unit.term, &analysis.live_out[b]);
    Ok(BlockCode {
        insts,
        term: unit.term,
        term_kills,
        entry_guards: unit.entry_guards,
        id,
    })
}
