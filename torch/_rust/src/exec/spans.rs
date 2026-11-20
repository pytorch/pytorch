// Finding spans (see cfg::ir::Span) in a code's Cfg as observed, before it's
// optimized: single-entry stretches of blocks whose every instr is movable
// (see observed_ops), reads or writes a local, or changes a container the
// stretch makes. Blocks never seen run may be in one (a trace that runs one
// gives up). A span runs as one region, which only replays what it computes
// and the locals it leaves, so the containers it changes must be its own.

use std::collections::{HashMap, HashSet};

use pyo3::prelude::*;

use super::effects::Effect;
use super::{makers, mutators, Prepared};
use crate::cfg::ir::*;
use crate::cfg::opcodes;

// A span in the Cfg it was found in (old block ids).
pub struct Found {
    pub entry: BlockId,
    pub exit: Option<BlockId>,
    pub never: Vec<BlockId>,
    pub live_in: Vec<u32>,
    pub live_out: Vec<u32>,
    pub inputs: Vec<(u32, Option<u32>)>,
}

fn read_slot(op: &Op) -> Option<u32> {
    match op {
        Op::LoadLocal { slot } | Op::LoadLocalChecked { slot } => Some(*slot),
        _ => None,
    }
}

fn written_slot(op: &Op) -> Option<u32> {
    match op {
        Op::StoreLocal { slot } | Op::DeleteLocal { slot } => Some(*slot),
        _ => None,
    }
}

// The locals live on entry to each of `blocks`, with `succs` giving the
// successors that count (others read nothing).
fn live_slots(cfg: &Cfg, blocks: &[BlockId], succs: impl Fn(BlockId) -> Vec<BlockId>) -> HashMap<BlockId, HashSet<u32>> {
    let mut uses: HashMap<BlockId, (HashSet<u32>, HashSet<u32>)> = HashMap::new();
    for b in blocks {
        let (mut used, mut defined) = (HashSet::new(), HashSet::new());
        for v in cfg[*b].versions.iter().flat_map(|ver| &ver.body) {
            if let Some(s) = read_slot(&cfg[*v].op)
                && !defined.contains(&s)
            {
                used.insert(s);
            }
            if let Some(s) = written_slot(&cfg[*v].op) {
                defined.insert(s);
            }
        }
        uses.insert(*b, (used, defined));
    }
    let mut live: HashMap<BlockId, HashSet<u32>> = blocks.iter().map(|b| (*b, HashSet::new())).collect();
    let mut changed = true;
    while changed {
        changed = false;
        for b in blocks.iter().rev() {
            let mut out: HashSet<u32> = HashSet::new();
            for s in succs(*b) {
                if let Some(l) = live.get(&s) {
                    out.extend(l);
                }
            }
            let (used, defined) = &uses[b];
            let mut l: HashSet<u32> = used.clone();
            l.extend(out.difference(defined));
            if l.len() != live[b].len() {
                live.insert(*b, l);
                changed = true;
            }
        }
    }
    live
}

// Immediate dominators over normal edges (Cooper, Harvey and Kennedy).
fn dominators(cfg: &Cfg) -> HashMap<BlockId, BlockId> {
    let order = cfg.rpo();
    let index: HashMap<BlockId, usize> = order.iter().enumerate().map(|(i, b)| (*b, i)).collect();
    let mut preds: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
    for b in &order {
        for s in cfg[*b].term.successors() {
            preds.entry(s).or_default().push(*b);
        }
    }
    let mut idom: HashMap<BlockId, BlockId> = HashMap::new();
    idom.insert(order[0], order[0]);
    let mut changed = true;
    while changed {
        changed = false;
        for b in order.iter().skip(1) {
            let mut new: Option<BlockId> = None;
            for p in preds.get(b).into_iter().flatten() {
                if !idom.contains_key(p) {
                    continue;
                }
                new = Some(match new {
                    None => *p,
                    Some(mut a) => {
                        let mut c = *p;
                        while a != c {
                            while index[&a] > index[&c] {
                                a = idom[&a];
                            }
                            while index[&c] > index[&a] {
                                c = idom[&c];
                            }
                        }
                        a
                    }
                });
            }
            if let Some(n) = new
                && idom.get(b) != Some(&n)
            {
                idom.insert(*b, n);
                changed = true;
            }
        }
    }
    idom
}

fn dominates(idom: &HashMap<BlockId, BlockId>, a: BlockId, mut b: BlockId) -> bool {
    loop {
        if a == b {
            return true;
        }
        match idom.get(&b) {
            Some(d) if *d != b => b = *d,
            _ => return false,
        }
    }
}

// The container an instr changes, if it only does that.
fn changed_container(py: Python<'_>, prep: &Prepared, v: ValueId) -> Option<ValueId> {
    let ins = &prep.cfg[v];
    let effects = unsafe { &*prep.effects.get() };
    let site = Site::Instr(v);
    if effects.varies(site) || effects.get(site).iter().any(|(seq, _)| seq.iter().any(|e| matches!(e, Effect::Frame { .. }))) {
        return None;
    }
    match ins.op {
        Op::Bytecode { opcode: opcodes::STORE_SUBSCR | opcodes::STORE_SLICE, .. } => Some(ins.args[1]),
        Op::Bytecode {
            opcode:
                opcodes::DELETE_SUBSCR
                | opcodes::LIST_APPEND
                | opcodes::SET_ADD
                | opcodes::MAP_ADD
                | opcodes::LIST_EXTEND
                | opcodes::SET_UPDATE
                | opcodes::DICT_UPDATE
                | opcodes::DICT_MERGE,
            ..
        } => Some(ins.args[0]),
        Op::Call { .. } if ins.args.len() > 1 => {
            let callees = unsafe { &*prep.callees.get() };
            let f = callees.get(&v).and_then(|c| c.get(py))?;
            mutators().contains(&(f.as_ptr() as usize)).then_some(ins.args[1])
        }
        _ => None,
    }
}

// Whether x is always a container made by an instr in `made` (the span's),
// not one held by a local the span may read before writing.
fn made_here(py: Python<'_>, prep: &Prepared, x: ValueId, made: &HashSet<ValueId>, live_in: &HashSet<u32>, seen: &mut HashSet<ValueId>) -> bool {
    let cfg = &prep.cfg;
    if !seen.insert(x) {
        return true;
    }
    match cfg[x].op {
        Op::Bytecode { opcode: opcodes::BUILD_LIST | opcodes::BUILD_MAP | opcodes::BUILD_SET, .. } => made.contains(&x),
        Op::Project(1) if matches!(cfg[cfg[x].args[0]].op, Op::LoadMethod(_)) => made_here(py, prep, cfg[cfg[x].args[0]].args[0], made, live_in, seen),
        Op::LoadLocal { slot } | Op::LoadLocalChecked { slot } => {
            let types = unsafe { &*prep.types.get() };
            // Stores of values never computed are on paths never taken.
            let stores: Vec<ValueId> = cfg
                .instrs
                .iter()
                .filter(|i| matches!(i.op, Op::StoreLocal { slot: s } if s == slot) && types.get(i.args[0].idx()).is_some_and(|t| t.ty.is_some()))
                .map(|i| i.args[0])
                .collect();
            !live_in.contains(&slot) && !stores.is_empty() && stores.iter().all(|s| made_here(py, prep, *s, made, live_in, seen))
        }
        Op::Call { .. } => {
            let callees = unsafe { &*prep.callees.get() };
            made.contains(&x) && callees.get(&x).and_then(|c| c.get(py)).is_some_and(|f| makers().contains(&(f.as_ptr() as usize)))
        }
        _ => false,
    }
}

pub fn find(py: Python<'_>, prep: &Prepared, movable: &HashSet<ValueId>) -> Vec<Found> {
    let cfg = &prep.cfg;
    let types = unsafe { &*prep.types.get() };
    let effects = unsafe { &*prep.effects.get() };
    let body = |b: BlockId| -> &[ValueId] { cfg[b].versions.first().map_or(&[], |v| v.body.as_slice()) };
    // Instrs whose value's type is observed whenever they run.
    let produces = |v: &ValueId| {
        matches!(
            cfg[*v].op,
            Op::LoadLocal { .. } | Op::LoadAttr(_) | Op::LoadMethod(_) | Op::LoadGlobal(_) | Op::LoadGlobalIn { .. } | Op::Call { .. } | Op::Binary(_) | Op::Compare(_) | Op::ToBool
        )
    };
    let never = |b: BlockId| body(b).iter().any(produces) && !body(b).iter().filter(|v| produces(v)).any(|v| types.get(v.idx()).is_some_and(|t| t.ty.is_some()));
    let frame_state = |v: &ValueId| {
        matches!(
            cfg[*v].op,
            Op::LoadDeref { .. } | Op::StoreDeref { .. } | Op::DeleteDeref { .. } | Op::MakeCell { .. } | Op::CopyFreeVars(_) | Op::LoadFromDictOrDeref { .. } | Op::Resume
        )
    };
    let ok = |b: BlockId| {
        let blk = &cfg[b];
        blk.versions.len() == 1
            && blk.origin.is_none()
            && matches!(blk.term, Terminator::Jump(Exit::Block(_) | Exit::Return(_)) | Terminator::Branch { .. })
            && !body(b).iter().any(frame_state)
            && (never(b)
                || body(b).iter().all(|v| {
                    movable.contains(v)
                        || matches!(cfg[*v].op, Op::LoadLocal { .. } | Op::LoadLocalChecked { .. } | Op::StoreLocal { .. } | Op::Project(_))
                        || changed_container(py, prep, *v).is_some()
                }))
    };
    let all: Vec<BlockId> = cfg.rpo();
    let frame_live = live_slots(cfg, &all, |b| cfg[b].successors());
    let idom = dominators(cfg);
    let mut used: HashSet<BlockId> = HashSet::new();
    let mut found = vec![];
    for &e in &all {
        if used.contains(&e) || e == ENTRY || !ok(e) || never(e) || !cfg.explicit_params(e).is_empty() {
            continue;
        }
        // A block never seen run is a dead end: a trace gives up on entering
        // it, so where it goes doesn't matter.
        let mut r = vec![e];
        let mut back_to_entry = false;
        let mut i = 0;
        while i < r.len() {
            let b = r[i];
            i += 1;
            if never(b) {
                continue;
            }
            for s in cfg[b].term.successors() {
                if s == e {
                    back_to_entry = true;
                } else if !r.contains(&s) && !used.contains(&s) && (ok(s) || never(s)) && dominates(&idom, e, s) {
                    r.push(s);
                }
            }
        }
        if back_to_entry {
            continue;
        }
        let in_r: HashSet<BlockId> = r.iter().copied().collect();
        let seen_r: Vec<BlockId> = r.iter().copied().filter(|b| !never(*b)).collect();
        let exits: HashSet<BlockId> = seen_r.iter().flat_map(|b| cfg[*b].term.successors()).filter(|s| !in_r.contains(s)).collect();
        let returns = seen_r.iter().any(|b| matches!(cfg[*b].term, Terminator::Jump(Exit::Return(_))));
        let exit = match (exits.len(), returns) {
            (1, false) => {
                let x = *exits.iter().next().unwrap();
                if !cfg[x].params.is_empty() {
                    continue;
                }
                Some(x)
            }
            (0, true) => None,
            _ => continue,
        };
        let seen_blocks = seen_r;
        // Worth one region: several instrs that ran torch ops or entered frames.
        let work = seen_blocks.iter().flat_map(|b| body(*b)).filter(|v| effects.get(Site::Instr(**v)).iter().any(|(seq, _)| !seq.is_empty())).count();
        if work < 4 || seen_blocks.len() < 2 {
            continue;
        }
        let seen_set: HashSet<BlockId> = seen_blocks.iter().copied().collect();
        let order: Vec<BlockId> = all.iter().copied().filter(|b| seen_set.contains(b)).collect();
        let succs = |b: BlockId| -> Vec<BlockId> { cfg[b].term.successors().into_iter().filter(|s| seen_set.contains(s)).collect() };
        let live = live_slots(cfg, &order, succs);
        let mut live_in: HashSet<u32> = live[&e].clone();
        let made: HashSet<ValueId> = r.iter().flat_map(|b| body(*b).iter().copied()).collect();
        let own = seen_blocks.iter().flat_map(|b| body(*b)).filter(|v| !movable.contains(v)).all(|v| match changed_container(py, prep, *v) {
            Some(x) => made_here(py, prep, x, &made, &live_in, &mut HashSet::new()),
            None => true,
        });
        if !own {
            continue;
        }
        let mut live_out: Vec<u32> = match exit {
            Some(x) => {
                let written: HashSet<u32> = seen_blocks.iter().flat_map(|b| body(*b)).filter_map(|v| written_slot(&cfg[*v].op)).collect();
                written.intersection(&frame_live[&x]).copied().collect()
            }
            None => vec![],
        };
        live_out.sort();
        // One left on some path as it was is also read (as an arg, so the
        // region gives it back rather than what the trace saw).
        let must = written_on_every_path(cfg, &order, e, exit, |b| body(b).iter().filter_map(|v| written_slot(&cfg[*v].op)).collect());
        live_in.extend(live_out.iter().filter(|s| !must.contains(s)));
        let mut live_in: Vec<u32> = live_in.into_iter().collect();
        live_in.sort();
        let inputs = inputs(cfg, &seen_blocks, &live_in, movable);
        used.extend(r.iter().copied());
        found.push(Found { entry: e, exit, never: r.iter().copied().filter(|b| never(*b)).collect(), live_in, live_out, inputs });
    }
    found
}

// The locals every path through `order` (RPO, from `entry`) writes before
// leaving to `exit` (or returning), with `defs` giving a block's writes.
fn written_on_every_path(cfg: &Cfg, order: &[BlockId], entry: BlockId, exit: Option<BlockId>, defs: impl Fn(BlockId) -> HashSet<u32>) -> HashSet<u32> {
    let within: HashSet<BlockId> = order.iter().copied().collect();
    let mut preds: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
    for b in order {
        for s in cfg[*b].term.successors() {
            preds.entry(s).or_default().push(*b);
        }
    }
    let mut out: HashMap<BlockId, HashSet<u32>> = HashMap::new();
    let mut changed = true;
    while changed {
        changed = false;
        for b in order {
            let mut at: Option<HashSet<u32>> = if *b == entry { Some(HashSet::new()) } else { None };
            for p in preds.get(b).into_iter().flatten().filter(|p| within.contains(p)) {
                if let Some(o) = out.get(p) {
                    at = Some(match at {
                        None => o.clone(),
                        Some(a) => a.intersection(o).copied().collect(),
                    });
                }
            }
            let Some(mut at) = at else { continue };
            at.extend(defs(*b));
            if out.get(b) != Some(&at) {
                out.insert(*b, at);
                changed = true;
            }
        }
    }
    let leaving = order.iter().filter(|b| match exit {
        Some(x) => cfg[**b].term.successors().contains(&x),
        None => matches!(cfg[**b].term, Terminator::Jump(Exit::Return(_))),
    });
    let mut must: Option<HashSet<u32>> = None;
    for b in leaving {
        let o = out.get(b).cloned().unwrap_or_default();
        must = Some(match must {
            None => o,
            Some(m) => m.intersection(&o).copied().collect(),
        });
    }
    must.unwrap_or_default()
}

// A span's args (see Span::inputs) for its `live_in` locals: the attributes
// read of one the blocks seen run only read movable attributes of (and never
// write), rather than the object (whose identity a region would check).
fn inputs(cfg: &Cfg, blocks: &[BlockId], live_in: &[u32], movable: &HashSet<ValueId>) -> Vec<(u32, Option<u32>)> {
    let vers: Vec<&Version> = blocks.iter().filter_map(|b| cfg[*b].versions.first()).collect();
    let instrs: Vec<ValueId> = vers.iter().flat_map(|v| v.body.iter().copied()).collect();
    let mut out = vec![];
    for &slot in live_in {
        let loads: HashSet<ValueId> = instrs.iter().copied().filter(|v| read_slot(&cfg[*v].op) == Some(slot)).collect();
        let written = instrs.iter().any(|v| written_slot(&cfg[*v].op) == Some(slot));
        let passed = vers.iter().any(|v| v.outputs.iter().any(|o| loads.contains(o)));
        let mut names: Vec<u32> = vec![];
        let attrs_only = !written
            && !passed
            && !loads.is_empty()
            && instrs.iter().all(|v| {
                let ins = &cfg[*v];
                match (&ins.op, ins.args.iter().position(|a| loads.contains(a))) {
                    (_, None) => true,
                    (&Op::LoadAttr(n), Some(0)) if ins.args.len() == 1 && movable.contains(v) => {
                        if !names.contains(&n) {
                            names.push(n);
                        }
                        true
                    }
                    _ => false,
                }
            });
        if attrs_only {
            names.sort();
            out.extend(names.into_iter().map(|n| (slot, Some(n))));
        } else {
            out.push((slot, None));
        }
    }
    out
}

// `found` in the Cfg optimized from the one it was found in (`renumbered`:
// each old block's new id).
pub fn place(cfg: &Cfg, found: Vec<Found>, renumbered: &[Option<BlockId>], qualname: &str) -> Vec<Span> {
    let mut spans = vec![];
    'spans: for f in found {
        let Some(entry) = renumbered[f.entry.idx()] else { continue };
        let exit = match f.exit {
            Some(x) => match renumbered[x.idx()] {
                Some(x) => Some(x),
                None => continue,
            },
            None => None,
        };
        let never: Vec<BlockId> = f.never.iter().filter_map(|b| renumbered[b.idx()]).collect();
        let mut blocks = vec![entry];
        let mut i = 0;
        while i < blocks.len() {
            let b = blocks[i];
            i += 1;
            if never.contains(&b) {
                continue;
            }
            if exit.is_some() && matches!(cfg[b].term, Terminator::Jump(Exit::Return(_))) {
                continue 'spans;
            }
            for s in cfg[b].term.successors() {
                if s == entry {
                    continue 'spans;
                }
                if Some(s) != exit && !blocks.contains(&s) {
                    blocks.push(s);
                }
            }
        }
        blocks.sort();
        spans.push(Span {
            entry,
            exit,
            blocks,
            never,
            live_in: f.live_in,
            live_out: f.live_out,
            inputs: f.inputs,
            name: format!("{qualname}:span bb{}", entry.0),
            state: Default::default(),
        });
    }
    spans
}
