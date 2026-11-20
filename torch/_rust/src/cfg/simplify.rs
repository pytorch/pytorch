// Cleanup passes run to a fixpoint after interpretation. Only explicit (Stack)
// params are ever removed; ENTRY has none. Values are local to a block, so a
// param can only be replaced by a reserved value.

use std::collections::{HashMap, HashSet};

use super::ir::*;

pub fn simplify(cfg: &mut Cfg) {
    loop {
        let mut changed = trivial_params(cfg);
        changed |= dead_params(cfg);
        changed |= cleanup_unwind(cfg);
        if !changed {
            break;
        }
    }
    cfg.compact_blocks();
}

// For every block, the args for its explicit params on each way in: one list
// per version of each predecessor edge, and one per handler segment.
fn incoming(cfg: &Cfg) -> Vec<Vec<Vec<ValueId>>> {
    let mut inc = vec![vec![]; cfg.blocks.len()];
    for b in cfg.block_ids() {
        let blk = &cfg[b];
        for e in blk.term.edges() {
            for ver in &blk.versions {
                inc[e.target.idx()].push(e.args.iter().map(|o| ver.outputs[*o as usize]).collect());
            }
        }
        for seg in blk.versions.iter().flat_map(|v| &v.unwind) {
            if let Handler::Block(k, args) = &seg.handler {
                inc[blk.handlers[*k].idx()].push(args.clone());
            }
        }
        if let Some(TermUnwind::Block(k, args)) = blk.term.unwind() {
            for ver in &blk.versions {
                inc[blk.handlers[*k].idx()].push(args.iter().map(|o| ver.outputs[*o as usize]).collect());
            }
        }
    }
    inc
}

fn substitute(cfg: &mut Cfg, map: &HashMap<ValueId, ValueId>) {
    if map.is_empty() {
        return;
    }
    let get = |v: &mut ValueId| {
        if let Some(n) = map.get(v) {
            *v = *n;
        }
    };
    let Cfg { instrs, blocks, .. } = cfg;
    for ver in blocks.iter_mut().filter(|b| !b.removed).flat_map(|b| &mut b.versions) {
        for v in &ver.body {
            instrs[v.idx()].args.iter_mut().for_each(get);
        }
        ver.outputs.iter_mut().for_each(get);
        for seg in &mut ver.unwind {
            if let Handler::Block(_, args) = &mut seg.handler {
                args.iter_mut().for_each(get);
            }
        }
    }
}

fn drop_positions<T>(args: &mut Vec<T>, positions: &[usize]) {
    for k in positions.iter().rev() {
        args.remove(*k);
    }
}

// Remove the explicit params of `b` at `positions` (ascending) together with
// the matching arg on every way in.
fn remove_params(cfg: &mut Cfg, b: BlockId, positions: &[usize]) {
    let explicit = cfg.explicit_params(b);
    let victims: HashSet<ValueId> = positions.iter().map(|k| explicit[*k]).collect();
    cfg[b].params.retain(|p| !victims.contains(p));
    for v in &victims {
        cfg[*v].op = Op::Removed;
    }
    let live: Vec<BlockId> = cfg.block_ids().collect();
    for src in live {
        let blk = &mut cfg[src];
        let mut touched = false;
        for e in blk.term.edges_mut().into_iter().filter(|e| e.target == b) {
            drop_positions(&mut e.args, positions);
            touched = true;
        }
        for seg in blk.versions.iter_mut().flat_map(|v| &mut v.unwind) {
            if let Handler::Block(k, args) = &mut seg.handler {
                if blk.handlers[*k] == b {
                    drop_positions(args, positions);
                }
            }
        }
        if let Some(TermUnwind::Block(k, args)) = blk.term.unwind_mut() {
            if blk.handlers[*k] == b {
                drop_positions(args, positions);
                touched = true;
            }
        }
        if touched {
            cfg.prune_outputs(src);
        }
    }
}

// A param whose every incoming arg is the same reserved value (or the param
// itself) is that value.
fn trivial_params(cfg: &mut Cfg) -> bool {
    let inc = incoming(cfg);
    let mut subst = HashMap::new();
    let mut removals = vec![];
    for b in cfg.block_ids().filter(|b| *b != ENTRY) {
        let mut remove = vec![];
        for (k, p) in cfg.explicit_params(b).iter().enumerate() {
            let mut args = inc[b.idx()].iter().map(|a| a[k]).filter(|a| a != p);
            let Some(v) = args.next() else { continue };
            if v.is_reserved() && args.all(|a| a == v) {
                remove.push(k);
                subst.insert(*p, v);
            }
        }
        if !remove.is_empty() {
            removals.push((b, remove));
        }
    }
    let changed = !removals.is_empty();
    for (b, remove) in removals {
        remove_params(cfg, b, &remove);
    }
    substitute(cfg, &subst);
    changed
}

fn dead_params(cfg: &mut Cfg) -> bool {
    let inc = incoming(cfg);
    // Explicit param -> (block, position).
    let mut where_param: HashMap<ValueId, (BlockId, usize)> = HashMap::new();
    for b in cfg.block_ids() {
        for (k, p) in cfg.explicit_params(b).iter().enumerate() {
            where_param.insert(*p, (b, k));
        }
    }
    let mut live = vec![false; cfg.instrs.len()];
    let mut work = vec![];
    let mark = |v: ValueId, live: &mut Vec<bool>, work: &mut Vec<ValueId>| {
        if !live[v.idx()] {
            live[v.idx()] = true;
            work.push(v);
        }
    };
    for b in cfg.block_ids() {
        let blk = &cfg[b];
        // Outputs read other than as args for params of this Cfg's blocks.
        let mut read = blk.term.outputs();
        let mut to_params: Vec<OutIdx> = blk.term.edges().iter().flat_map(|e| e.args.clone()).collect();
        if let Some(TermUnwind::Block(_, args)) = blk.term.unwind() {
            to_params.extend(args);
        }
        for o in to_params {
            let k = read.iter().position(|r| *r == o).expect("an arg is an output the terminator reads");
            read.swap_remove(k);
        }
        for ver in &blk.versions {
            for v in &ver.body {
                for a in &cfg[*v].args {
                    mark(*a, &mut live, &mut work);
                }
            }
            for o in &read {
                mark(ver.outputs[*o as usize], &mut live, &mut work);
            }
        }
    }
    while let Some(v) = work.pop() {
        if let Some((b, k)) = where_param.get(&v) {
            for args in &inc[b.idx()] {
                mark(args[*k], &mut live, &mut work);
            }
        }
    }
    let mut changed = false;
    let blocks: Vec<BlockId> = cfg.block_ids().filter(|b| *b != ENTRY).collect();
    for b in blocks {
        let dead: Vec<usize> = cfg
            .explicit_params(b)
            .iter()
            .enumerate()
            .filter(|(_, p)| !live[p.idx()])
            .map(|(k, _)| k)
            .collect();
        if !dead.is_empty() {
            remove_params(cfg, b, &dead);
            changed = true;
        }
    }
    changed
}

fn cleanup_unwind(cfg: &mut Cfg) -> bool {
    let mut changed = false;
    let blocks: Vec<BlockId> = cfg.block_ids().collect();
    for b in &blocks {
        let before: Vec<usize> = cfg[*b].versions.iter().map(|v| v.unwind.len()).collect();
        cfg.normalize_unwind(*b);
        changed |= cfg[*b].versions.iter().map(|v| v.unwind.len()).ne(before);
    }
    let reach = cfg.reachable();
    for b in blocks {
        if !reach[b.idx()] {
            cfg.remove_block(b);
            changed = true;
        }
    }
    changed
}
