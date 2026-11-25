// Outlining: replaces a region of blocks with one block that keeps the region
// as its origin, a Cfg of its own run when no version's guard holds. For now
// the regions are chains (each block jumps to the next, whose only way in that
// is), merged into one unguarded version, or into no version at all so the
// origin always runs (which exercises the miss path).

use std::collections::HashMap;
use std::sync::Arc;

use super::ir::*;
use super::opcodes;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Outline {
    // Each chain becomes one block with a merged version.
    Merge,
    // Each chain becomes one block with only its origin.
    OriginOnly,
}

// A generator suspends inside one block's version and resumes there; keep
// those blocks as they are.
pub fn outlinable(cfg: &Cfg, b: BlockId) -> bool {
    let blk = &cfg[b];
    let resume_point = |v: &ValueId| {
        matches!(
            cfg[*v].op,
            Op::Yield
                | Op::Bytecode {
                    opcode: opcodes::RETURN_GENERATOR,
                    ..
                }
        )
    };
    blk.origin.is_none() && blk.versions.len() == 1 && !blk.versions[0].body.iter().any(resume_point)
}

pub fn outline_chains(cfg: &mut Cfg, mode: Outline) {
    let mut preds = vec![0usize; cfg.blocks.len()];
    for b in cfg.block_ids() {
        for s in cfg[b].successors() {
            preds[s.idx()] += 1;
        }
    }
    let mut taken = vec![false; cfg.blocks.len()];
    for a in cfg.block_ids().collect::<Vec<_>>() {
        if taken[a.idx()] || !outlinable(cfg, a) {
            continue;
        }
        let mut chain = vec![a];
        while let Terminator::Jump(Exit::Block(e)) = &cfg[*chain.last().unwrap()].term {
            let t = e.target;
            if t == ENTRY || preds[t.idx()] != 1 || taken[t.idx()] || chain.contains(&t) || !outlinable(cfg, t) {
                break;
            }
            chain.push(t);
        }
        if chain.len() > 1 {
            for b in &chain {
                taken[b.idx()] = true;
            }
            outline(cfg, &chain, mode);
        }
    }
    cfg.compact_blocks();
}

// Replaces the chain with its head block.
pub fn outline(cfg: &mut Cfg, chain: &[BlockId], mode: Outline) {
    let mut handlers: Vec<BlockId> = vec![];
    for b in chain {
        for h in &cfg[*b].handlers {
            if !handlers.contains(h) {
                handlers.push(*h);
            }
        }
    }
    let origin = extract(cfg, chain, &handlers);
    let handler_of = |blk: &Block, k: usize| handlers.iter().position(|h| *h == blk.handlers[k]).unwrap();

    // The merged version: the bodies in order, each block's params replaced by
    // what the previous one passes.
    let mut merged = Version::default();
    let mut subst: HashMap<ValueId, ValueId> = HashMap::new();
    let get = |subst: &HashMap<ValueId, ValueId>, v: ValueId| subst.get(&v).copied().unwrap_or(v);
    let last = *chain.last().unwrap();
    for (i, b) in chain.iter().enumerate() {
        let blk = cfg[*b].clone();
        let ver = &blk.versions[0];
        let off = merged.body.len();
        // Ops before the block's first segment propagate.
        if ver.unwind.first().is_none_or(|s| s.from > 0) {
            merged.unwind.push(UnwindSegment {
                from: off,
                handler: Handler::Propagate,
            });
        }
        for seg in &ver.unwind {
            let handler = match &seg.handler {
                Handler::Block(k, args) => {
                    Handler::Block(handler_of(&blk, *k), args.iter().map(|a| get(&subst, *a)).collect())
                }
                h => h.clone(),
            };
            merged.unwind.push(UnwindSegment {
                from: seg.from + off,
                handler,
            });
        }
        for v in &ver.body {
            let args = cfg[*v].args.iter().map(|a| get(&subst, *a)).collect();
            cfg[*v].args = args;
        }
        merged.body.extend(&ver.body);
        let outputs: Vec<ValueId> = ver.outputs.iter().map(|o| get(&subst, *o)).collect();
        match chain.get(i + 1) {
            Some(next) => {
                let Terminator::Jump(Exit::Block(e)) = &blk.term else {
                    unreachable!("chains are linked by jumps")
                };
                for (k, p) in cfg.explicit_params(*next).iter().enumerate() {
                    subst.insert(*p, outputs[e.args[k] as usize]);
                }
            }
            None => merged.outputs = outputs,
        }
    }

    let head = chain[0];
    let (mut term, term_origin) = (cfg[last].term.clone(), cfg[last].term_origin);
    if let Some(TermUnwind::Block(k, _)) = term.unwind_mut() {
        *k = handler_of(&cfg[last], *k);
    }
    for b in &chain[1..] {
        let blk = std::mem::replace(&mut cfg[*b], Block::new());
        for p in &blk.params {
            cfg[*p].op = Op::Removed;
        }
        cfg[*b].removed = true;
    }
    if mode == Outline::OriginOnly {
        for v in &merged.body {
            cfg[*v].op = Op::Removed;
            cfg[*v].args.clear();
        }
    }
    let blk = &mut cfg[head];
    blk.versions = match mode {
        Outline::Merge => vec![merged],
        Outline::OriginOnly => vec![],
    };
    blk.term = term;
    blk.term_origin = term_origin;
    blk.handlers = handlers;
    blk.origin = Some(Arc::new(origin));
    cfg.normalize_unwind(head);
}

// A copy of the chain as a Cfg of its own: its exit leaves it with all the
// last block's outputs (the owner's terminator, the last block's, then
// dispatches on them, raising if it raises), and raises into the chain's
// handlers go to the owner's.
fn extract(cfg: &Cfg, chain: &[BlockId], handlers: &[BlockId]) -> Cfg {
    let mut sub = Cfg::new(cfg.slot_names.clone(), cfg.const_reprs.clone(), cfg.names.clone());
    sub.literals = cfg.literals.clone();
    sub.trainings = cfg.trainings.clone();
    let last = &cfg[*chain.last().unwrap()];
    sub.boundary = Some(Boundary {
        n_outputs: last.n_outputs(),
        handler_arity: handlers.iter().map(|h| cfg.explicit_params(*h).len()).collect(),
    });
    // The reserved values have the same ids in every Cfg.
    let mut map: HashMap<ValueId, ValueId> = HashMap::new();
    let get = |map: &HashMap<ValueId, ValueId>, v: ValueId| if v.is_reserved() { v } else { map[&v] };
    for (i, b) in chain.iter().enumerate() {
        let blk = &cfg[*b];
        let ver = &blk.versions[0];
        let mut nb = Block::new();
        for p in &blk.params {
            let np = sub.add(cfg[*p].op.clone(), vec![], cfg[*p].origin);
            map.insert(*p, np);
            nb.params.push(np);
        }
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
        nb.term = match (&blk.term, i + 1 < chain.len()) {
            (Terminator::Jump(Exit::Block(e)), true) => Terminator::Jump(Exit::Block(Edge {
                target: BlockId(i as u32 + 1),
                args: e.args.clone(),
            })),
            (_, false) => Terminator::Jump(Exit::Leave((0..blk.n_outputs() as OutIdx).collect())),
            _ => unreachable!("chains are linked by jumps"),
        };
        nb.term_origin = blk.term_origin;
        sub.blocks.push(nb);
    }
    sub
}
