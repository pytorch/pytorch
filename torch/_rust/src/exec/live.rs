// Liveness of SSA values within each version, so the executor can drop each
// reference after its last use (CPython drops a stack value when it's popped).
// Values are local to a block, so nothing is live across blocks: leaving a
// block drops whatever of it is left. Locals need nothing: the frame owns them.
//
// An instr that can raise also "uses" its segment's args.

use std::collections::HashSet;

use crate::cfg::ir::*;

pub struct VersionLive {
    // Params the version doesn't use, dropped when it starts.
    pub entry_kill: Vec<ValueId>,
    // Values dropped after body[i] executes.
    pub after_kill: Vec<Vec<ValueId>>,
}

pub struct Liveness {
    // By block, then version.
    pub versions: Vec<Vec<VersionLive>>,
    // A Yield after which a value other than its stack args is live (the
    // executor saves only the stack in a suspended frame).
    pub bad_yield: Option<ValueId>,
}

// The segment covering body[at], if any.
pub fn segment_at(ver: &Version, at: usize) -> Option<&UnwindSegment> {
    ver.unwind.iter().rev().find(|s| s.from <= at)
}

fn uses(live: &mut HashSet<ValueId>, vs: &[ValueId]) {
    live.extend(vs.iter().filter(|v| !v.is_reserved()));
}

fn version(cfg: &Cfg, blk: &Block, ver: &Version, bad_yield: &mut Option<ValueId>) -> VersionLive {
    let mut live = HashSet::new();
    uses(&mut live, &ver.outputs);
    let mut after_kill = vec![vec![]; ver.body.len()];
    for i in (0..ver.body.len()).rev() {
        let v = ver.body[i];
        let ins = &cfg[v];
        let mut candidates: Vec<ValueId> = ins.args.iter().copied().filter(|a| !a.is_reserved()).collect();
        candidates.push(v);
        candidates.sort();
        candidates.dedup();
        after_kill[i] = candidates.into_iter().filter(|c| !live.contains(c)).collect();
        if ins.op == Op::Yield && live.iter().any(|l| *l != v && !ins.args.contains(l)) {
            *bad_yield = Some(v);
        }
        live.remove(&v);
        uses(&mut live, &ins.args);
        if ins.op.can_raise() {
            if let Some(Handler::Block(_, args) | Handler::Outer(_, args)) = segment_at(ver, i).map(|s| &s.handler) {
                uses(&mut live, args);
            }
        }
    }
    VersionLive {
        entry_kill: blk.params.iter().copied().filter(|p| !live.contains(p)).collect(),
        after_kill,
    }
}

pub fn compute(cfg: &Cfg) -> Liveness {
    let mut bad_yield = None;
    let versions = cfg
        .blocks
        .iter()
        .map(|blk| blk.versions.iter().map(|ver| version(cfg, blk, ver, &mut bad_yield)).collect())
        .collect();
    Liveness { versions, bad_yield }
}
