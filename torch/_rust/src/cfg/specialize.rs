// Specialization: replaces a region whose branches each went one way when
// observed with one block whose version runs the path taken. Each branch
// becomes a check in the version's guard (see guard.rs), so its condition
// must be known on entry: computed from the frame's locals and
// what reads can be moved ahead of the region's other code. A failing check
// runs the block's origin, a copy of the whole region, from its start.
//
// A region is single-entry: its head, then every block reachable from it up
// to its exit block (the last on the path), all of whose predecessors are in
// the region. Its other blocks may leave it only the way the exit block does
// (a return, or a jump to the same block), or by raising out of the frame.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::ir::*;
use super::outline::outlinable;

// Specializes the region starting at `head` along the directions `dirs` its
// branches were seen to go, extending the path as far as it stays a region.
// Returns each old block's new id (blocks are renumbered).
// `exact`: for a ToBool or Compare, a check on each arg (an exact builtin
// type) under which it runs no Python code, so it may move ahead of calls.
pub fn specialize(cfg: &mut Cfg, head: BlockId, dirs: &HashMap<BlockId, bool>, exact: &HashMap<ValueId, Vec<Check>>) -> Result<Vec<Option<BlockId>>, String> {
    if head.idx() >= cfg.blocks.len() || !outlinable(cfg, head) {
        return Err(format!("bb{} can't be specialized: it already has an origin or several versions, or a generator suspends in it", head.0));
    }
    let mut path = vec![head];
    loop {
        let b = *path.last().unwrap();
        let next = match (&cfg[b].term, dirs.get(&b)) {
            (Terminator::Jump(Exit::Block(e)), _) => e.target,
            (Terminator::Branch { then, else_, .. }, Some(d)) => match if *d { then } else { else_ } {
                Exit::Block(e) => e.target,
                _ => break,
            },
            _ => break,
        };
        if next == ENTRY || path.contains(&next) || !outlinable(cfg, next) {
            break;
        }
        path.push(next);
    }
    for end in (1..path.len()).rev() {
        if !path[..end].iter().any(|b| matches!(cfg[*b].term, Terminator::Branch { .. })) {
            break;
        }
        if let Some(region) = region(cfg, head, path[end]) {
            let mut c = cfg.clone();
            if build(&mut c, &path[..=end], &region, dirs, exact) {
                *cfg = c;
                return Ok(cfg.compact_blocks());
            }
        }
    }
    Err(format!("bb{}: no region from there has its branches seen going one way, with conditions known on entry", head.0))
}

// The exit block's exit, if the region's other blocks may leave the same way:
// x's terminator is a jump.
fn x_exit(cfg: &Cfg, x: BlockId) -> Option<&Exit> {
    match &cfg[x].term {
        Terminator::Jump(e) => Some(e),
        _ => None,
    }
}

// Whether `e`, from a region block other than x, leaves the region as x does.
fn same_exit(cfg: &Cfg, x: BlockId, e: &Exit) -> bool {
    match (x_exit(cfg, x), e) {
        (Some(Exit::Return(_)), Exit::Return(_)) => true,
        (Some(Exit::Block(a)), Exit::Block(b)) => a.target == b.target,
        _ => false,
    }
}

// The region from head to x (head first), if it is one.
fn region(cfg: &Cfg, head: BlockId, x: BlockId) -> Option<Vec<BlockId>> {
    let mut r = vec![head];
    let mut i = 0;
    while i < r.len() {
        let b = r[i];
        i += 1;
        if b == x {
            continue;
        }
        if !outlinable(cfg, b) || cfg[b].term.unwind().is_some_and(|u| *u != TermUnwind::Propagate) {
            return None;
        }
        for e in cfg[b].term.exits() {
            match e {
                _ if same_exit(cfg, x, e) => {}
                Exit::Block(edge) => {
                    let t = edge.target;
                    if t == ENTRY || cfg.explicit_params(t).len() != cfg[t].params.len() {
                        return None;
                    }
                    if !r.contains(&t) {
                        r.push(t);
                    }
                }
                _ => return None,
            }
        }
    }
    if !r.contains(&x) {
        return None;
    }
    if cfg[x].term.successors().iter().any(|t| r.contains(t)) {
        return None;
    }
    for p in cfg.block_ids() {
        for s in cfg[p].successors() {
            let inside = r.contains(&p);
            if r.contains(&s) && (s == head) == inside {
                return None;
            }
        }
    }
    Some(r)
}

// The outputs x's terminator reads, in order: the owner block's outputs.
fn used(cfg: &Cfg, x: BlockId) -> Vec<OutIdx> {
    let mut u = vec![];
    for o in cfg[x].term.outputs() {
        if !u.contains(&o) {
            u.push(o);
        }
    }
    u
}

// `e` (from a region block other than x, leaving as x does) as a Leave giving
// the owner block's outputs (see used).
fn leave(cfg: &Cfg, x: BlockId, e: &Exit) -> Exit {
    let (want, have): (Vec<OutIdx>, Vec<OutIdx>) = match (x_exit(cfg, x), e) {
        (Some(Exit::Return(a)), Exit::Return(b)) => (vec![*a], vec![*b]),
        (Some(Exit::Block(a)), Exit::Block(b)) => (a.args.clone(), b.args.clone()),
        _ => unreachable!("checked by same_exit"),
    };
    Exit::Leave(used(cfg, x).iter().map(|u| have[want.iter().position(|w| w == u).unwrap()]).collect())
}

// Whether it built the region (false: a check didn't make it into the
// guard).
fn build(cfg: &mut Cfg, path: &[BlockId], region: &[BlockId], dirs: &HashMap<BlockId, bool>, exact: &HashMap<ValueId, Vec<Check>>) -> bool {
    let head = path[0];
    let x = *path.last().unwrap();
    let mut handlers: Vec<BlockId> = vec![];
    for b in region {
        for h in &cfg[*b].handlers {
            if !handlers.contains(h) {
                handlers.push(*h);
            }
        }
    }
    let handler_of = |blk: &Block, k: usize| handlers.iter().position(|h| *h == blk.handlers[k]).unwrap();
    let mut origin = extract(cfg, region, x, &handlers);
    origin.compact_blocks();

    // The merged version: the path's bodies in order, each block's params
    // replaced by what the previous one passes, a check at each branch (and
    // before a ToBool or Compare `exact` has checks for). Each instr's handler, by
    // instr.
    let mut merged = Version::default();
    let mut raises_to: HashMap<ValueId, Handler> = HashMap::new();
    let mut subst: HashMap<ValueId, ValueId> = HashMap::new();
    let get = |subst: &HashMap<ValueId, ValueId>, v: ValueId| subst.get(&v).copied().unwrap_or(v);
    for (i, b) in path.iter().enumerate() {
        let blk = cfg[*b].clone();
        let ver = &blk.versions[0];
        for (k, v) in ver.body.iter().enumerate() {
            let args: Vec<ValueId> = cfg[*v].args.iter().map(|a| get(&subst, *a)).collect();
            cfg[*v].args = args.clone();
            for (c, a) in exact.get(v).into_iter().flatten().zip(&args) {
                let check = cfg.add(Op::expect(c.clone()), vec![*a], cfg[*v].origin);
                merged.body.push(check);
            }
            let handler = match ver.handler_at(k) {
                Handler::Block(k, args) => Handler::Block(handler_of(&blk, k), args.iter().map(|a| get(&subst, *a)).collect()),
                h => h,
            };
            raises_to.insert(*v, handler);
            merged.body.push(*v);
        }
        let outputs: Vec<ValueId> = ver.outputs.iter().map(|o| get(&subst, *o)).collect();
        let Some(next) = path.get(i + 1) else {
            merged.outputs = used(cfg, x).iter().map(|o| outputs[*o as usize]).collect();
            break;
        };
        let taken = match &blk.term {
            Terminator::Jump(Exit::Block(e)) => e,
            Terminator::Branch { cond, then, else_ } => {
                let d = dirs[b];
                let check = cfg.add(Op::expect(Check::Value(Lit::Bool(d))), vec![outputs[*cond as usize]], blk.term_origin);
                merged.body.push(check);
                match if d { then } else { else_ } {
                    Exit::Block(e) => e,
                    _ => unreachable!("the path follows edges"),
                }
            }
            _ => unreachable!("the path follows edges"),
        };
        for (k, p) in cfg.explicit_params(*next).iter().enumerate() {
            subst.insert(*p, outputs[taken.args[k] as usize]);
        }
    }

    let (mut term, term_origin) = (cfg[x].term.clone(), cfg[x].term_origin);
    if let Some(TermUnwind::Block(k, _)) = term.unwind_mut() {
        *k = handler_of(&cfg[x], *k);
    }
    let u = used(cfg, x);
    for o in term.outputs_mut() {
        *o = u.iter().position(|w| w == o).unwrap() as OutIdx;
    }
    for b in region.iter().filter(|b| **b != head) {
        if path.contains(b) {
            let blk = std::mem::replace(&mut cfg[*b], Block::new());
            for p in &blk.params {
                cfg[*p].op = Op::Removed;
            }
            cfg[*b].removed = true;
        } else {
            cfg.remove_block(*b);
        }
    }
    let blk = &mut cfg[head];
    blk.versions = vec![merged];
    blk.term = term;
    blk.term_origin = term_origin;
    blk.handlers = handlers;
    blk.origin = Some(Arc::new(origin));

    // Reads move into the guard unless what ran before them may have changed
    // what they read: a local stored (or deleted) before, or any other read
    // after code that can run Python (a call, an operator).
    let (mut written, mut all_written, mut opaque) = (HashSet::new(), false, false);
    let ok = super::guard::lift_guard(cfg, head, |cfg, x| {
        let stale = match cfg[x].op {
            Op::LoadLocal { slot } => all_written || written.contains(&slot),
            Op::ToBool | Op::Compare(_) => opaque && !exact.contains_key(&x),
            Op::LoadGlobal(_) | Op::LoadGlobalIn { .. } | Op::LoadAttr(_) | Op::LoadMethod(_) | Op::Call { .. } | Op::Bytecode { .. } | Op::Resume => opaque,
            _ => false,
        };
        match cfg[x].op {
            Op::StoreLocal { slot } | Op::DeleteLocal { slot } | Op::MakeCell { slot } => {
                written.insert(slot);
            }
            Op::CopyFreeVars(_) => all_written = true,
            _ => {}
        }
        opaque |= stale || !super::inline::is_read(cfg, x);
        stale
    }, |v| raises_to.get(&v).cloned().unwrap_or(Handler::Propagate));
    ok
}

// A copy of the region as a Cfg of its own: x leaves it with all its outputs
// (the owner's terminator, x's, then dispatches on them), as do the other
// ways out x's terminator shares; raises into the region's handlers go to the
// owner's.
fn extract(cfg: &Cfg, region: &[BlockId], x: BlockId, handlers: &[BlockId]) -> Cfg {
    let mut sub = Cfg::new(cfg.slot_names.clone(), cfg.const_reprs.clone(), cfg.names.clone());
    sub.literals = cfg.literals.clone();
    sub.trainings = cfg.trainings.clone();
    sub.boundary = Some(Boundary {
        n_outputs: used(cfg, x).len(),
        handler_arity: handlers.iter().map(|h| cfg.explicit_params(*h).len()).collect(),
    });
    let index = |b: BlockId| region.iter().position(|r| *r == b);
    let mut map: HashMap<ValueId, ValueId> = HashMap::new();
    let get = |map: &HashMap<ValueId, ValueId>, v: ValueId| if v.is_reserved() { v } else { map[&v] };
    for b in region {
        let blk = &cfg[*b];
        let mut nb = Block::new();
        for p in &blk.params {
            let np = sub.add(cfg[*p].op.clone(), vec![], cfg[*p].origin);
            map.insert(*p, np);
            nb.params.push(np);
        }
        let ver = &blk.versions[0];
        let mut nver = Version::default();
        for v in &ver.body {
            let ins = &cfg[*v];
            let args = ins.args.iter().map(|a| get(&map, *a)).collect();
            let nv = sub.add(ins.op.clone(), args, ins.origin);
            map.insert(*v, nv);
            nver.body.push(nv);
        }
        nver.outputs = ver.outputs.iter().map(|o| get(&map, *o)).collect();
        nver.unwind = ver
            .unwind
            .iter()
            .map(|seg| UnwindSegment {
                from: seg.from,
                handler: match &seg.handler {
                    Handler::Block(k, args) => Handler::Outer(
                        handlers.iter().position(|h| *h == blk.handlers[*k]).unwrap(),
                        args.iter().map(|a| get(&map, *a)).collect(),
                    ),
                    h => h.clone(),
                },
            })
            .collect();
        nb.versions.push(nver);
        nb.term = if *b == x {
            Terminator::Jump(Exit::Leave(used(cfg, x)))
        } else {
            let mut term = blk.term.clone();
            let exits: Vec<&mut Exit> = match &mut term {
                Terminator::Jump(e) => vec![e],
                Terminator::Branch { then, else_, .. } => vec![then, else_],
                _ => vec![],
            };
            for e in exits {
                *e = if same_exit(cfg, x, e) {
                    leave(cfg, x, e)
                } else {
                    match &*e {
                        Exit::Block(edge) => Exit::Block(Edge { target: BlockId(index(edge.target).unwrap() as u32), args: edge.args.clone() }),
                        _ => unreachable!("checked by region"),
                    }
                };
            }
            term
        };
        nb.term_origin = blk.term_origin;
        sub.blocks.push(nb);
    }
    sub
}
