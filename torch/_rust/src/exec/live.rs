// Liveness of SSA values within each version, so the executor can drop each
// reference after its last use (CPython drops a stack value when it's popped).
// Values are local to a block, so nothing is live across blocks: leaving a
// block drops whatever of it is left. Locals need nothing: the frame owns them.
//
// An instr that can raise also "uses" its segment's args.
//
// THE INVARIANT of an instr that owns its operands (order.rs `owns`): it is
// handed as its own (moved to it, run.rs) only an operand that nothing reads
// behind it: no later instr, no way out of the version (an output), and no
// handler its step can unwind to. Every other operand it shares: it gets a
// reference of its own and the version keeps the one it has. `taken` is the
// one place that says which; `read_behind` holds what it said against the
// plan itself, and a plan that fails there is not run (its code is left to
// CPython, exec/mod.rs).

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};

use pyo3::prelude::*;

use crate::cfg::ir::*;

pub struct VersionLive {
    // Params the version doesn't use, dropped when it starts, the top one
    // first (order.rs).
    pub entry_kill: Vec<ValueId>,
    // Values dropped after body[i] executes: its operands in the order the
    // instruction lets go of them (order.rs), then its own value if nothing
    // reads it.
    pub after_kill: Vec<Vec<ValueId>>,
    // By place, for each instr of the body that owns its operands: those
    // it takes with it (THE INVARIANT).
    pub takes: Vec<(usize, Vec<ValueId>)>,
}

impl VersionLive {
    // What body[i] takes with it: nothing, if it does not own its operands.
    pub fn taken(&self, i: usize) -> &[ValueId] {
        self.takes.binary_search_by_key(&i, |(at, _)| *at).map_or(&[], |k| &self.takes[k].1)
    }
}

pub struct Liveness {
    // By block, then version.
    pub versions: Vec<Vec<VersionLive>>,
    // A Yield after which a value other than its stack args is live (the
    // executor saves only the stack in a suspended frame).
    pub bad_yield: Option<ValueId>,
    // An instr whose op has no order in which it lets go of its operands.
    pub no_order: Option<ValueId>,
    // An instr that would take with it an operand that is read behind it
    // (`read_behind`).
    pub takes_read: Option<ValueId>,
}

// The segment covering body[at], if any.
pub fn segment_at(ver: &Version, at: usize) -> Option<&UnwindSegment> {
    ver.unwind.iter().rev().find(|s| s.from <= at)
}

// What the segment covering body[at] hands its handler.
pub fn handed_at(ver: &Version, at: usize) -> &[ValueId] {
    match segment_at(ver, at).map(|s| &s.handler) {
        Some(Handler::Block(_, args) | Handler::Outer(_, args)) => args,
        _ => &[],
    }
}

fn uses(live: &mut HashSet<ValueId>, vs: &[ValueId]) {
    live.extend(vs.iter().filter(|v| !v.is_reserved()));
}

// For tests of `read_behind`: an instr that owns its operands is taken to
// take every operand its step drops, read behind it or not.
static TAKES_ALL: AtomicBool = AtomicBool::new(false);

/// For tests: with `on`, the plans made from now on hand an instruction
/// that owns its operands every operand its step drops, whether something
/// reads it behind the instruction or not (False ends it). A plan in which
/// that hands over an operand that is read is not run: its code is left to
/// CPython.
#[pyfunction]
pub fn _cfg_takes_all(on: bool) {
    TAKES_ALL.store(on, Ordering::Relaxed);
}

// THE INVARIANT, said once: of the operands of `ins`, which owns them,
// those it takes with it. `dead`: what no instr or way out behind it reads;
// `handed`: what the handler its step can unwind to is handed.
fn taken(ins: &Instr, dead: &[ValueId], handed: &[ValueId]) -> Vec<ValueId> {
    ins.args.iter().copied().filter(|a| dead.contains(a) && !handed.contains(a)).collect()
}

// THE INVARIANT, held against the plan itself and not against what the
// pass above made of it: Some(the instr) for an instr of the version that
// would take with it an operand that an instr behind it reads, or a
// handler from its own place on, or a way out.
fn read_behind(cfg: &Cfg, ver: &Version, lv: &VersionLive) -> Option<ValueId> {
    for (i, takes) in &lv.takes {
        let behind = &ver.body[*i + 1..];
        let read = |x: &ValueId| {
            behind.iter().any(|v| cfg[*v].args.contains(x))
                || (*i..ver.body.len()).any(|k| handed_at(ver, k).contains(x))
                || ver.outputs.contains(x)
        };
        if takes.iter().any(|x| x.is_reserved() || !cfg[ver.body[*i]].args.contains(x) || read(x)) {
            return Some(ver.body[*i]);
        }
    }
    None
}

fn version(cfg: &Cfg, blk: &Block, ver: &Version, bad_yield: &mut Option<ValueId>, no_order: &mut Option<ValueId>) -> VersionLive {
    let mut live = HashSet::new();
    uses(&mut live, &ver.outputs);
    let mut after_kill = vec![vec![]; ver.body.len()];
    let mut takes = vec![];
    for i in (0..ver.body.len()).rev() {
        let v = ver.body[i];
        let ins = &cfg[v];
        let handed = handed_at(ver, i);
        let mut candidates = super::order::releases(&ins.op, &ins.args).unwrap_or_else(|| {
            *no_order = Some(v);
            vec![]
        });
        candidates.push(v);
        // `live`: what is read behind body[i], by an instr or its handler,
        // by a way out of the version.
        after_kill[i] = candidates.into_iter().filter(|c| !live.contains(c)).collect();
        if super::order::owns(&ins.op) {
            let all = TAKES_ALL.load(Ordering::Relaxed);
            takes.push((i, taken(ins, &after_kill[i], if all { &[] } else { handed })));
        }
        if ins.op == Op::Yield && live.iter().any(|l| *l != v && !ins.args.contains(l)) {
            *bad_yield = Some(v);
        }
        live.remove(&v);
        uses(&mut live, &ins.args);
        if ins.op.can_raise() {
            uses(&mut live, handed);
        }
    }
    takes.reverse();
    VersionLive {
        entry_kill: blk.params.iter().rev().copied().filter(|p| !live.contains(p)).collect(),
        after_kill,
        takes,
    }
}

pub fn compute(cfg: &Cfg) -> Liveness {
    let (mut bad_yield, mut no_order) = (None, None);
    let versions: Vec<Vec<VersionLive>> = cfg
        .blocks
        .iter()
        .map(|blk| blk.versions.iter().map(|ver| version(cfg, blk, ver, &mut bad_yield, &mut no_order)).collect())
        .collect();
    let takes_read = cfg
        .blocks
        .iter()
        .zip(&versions)
        .flat_map(|(blk, lvs)| blk.versions.iter().zip(lvs))
        .find_map(|(ver, lv)| read_behind(cfg, ver, lv));
    Liveness { versions, bad_yield, no_order, takes_read }
}
