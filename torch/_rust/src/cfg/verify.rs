// Structural and SSA checks. Returns Err (never panics); a failure is an
// internal bug, surfaced in Python as RuntimeError.

use std::collections::HashSet;

use super::ir::*;
use super::opcodes;

pub fn verify(cfg: &Cfg) -> Result<(), String> {
    let nb = cfg.blocks.len();
    if nb == 0 {
        return Err("no blocks".into());
    }
    let order = cfg.rpo();
    if order.len() != nb || order.iter().enumerate().any(|(i, b)| b.idx() != i) {
        return Err(format!("blocks are not numbered in RPO (or unreachable): {order:?}"));
    }

    let mut preds = vec![0usize; nb];
    for (i, blk) in cfg.blocks.iter().enumerate() {
        if blk.removed {
            return Err(format!("bb{i} is removed"));
        }
        if matches!(blk.term, Terminator::Unreachable) {
            return Err(format!("bb{i} has an unreachable terminator"));
        }
        for s in blk.successors() {
            if s.idx() >= nb {
                return Err(format!("bb{i} has dangling successor bb{}", s.0));
            }
            preds[s.idx()] += 1;
        }
    }
    if preds[ENTRY.idx()] != 0 {
        return Err("ENTRY has predecessors".into());
    }
    // An outlined Cfg's ENTRY takes its owner's params.
    if cfg.boundary.is_none() && !cfg[ENTRY].params.is_empty() {
        return Err("ENTRY has params".into());
    }

    // Ids: each in at most one block, params are Params, instrs are not.
    let mut seen = HashSet::new();
    for (i, blk) in cfg.blocks.iter().enumerate() {
        for p in &blk.params {
            if !matches!(cfg[*p].op, Op::Param(_)) || p.is_reserved() || !seen.insert(*p) {
                return Err(format!("bb{i}: bad or duplicate param %{}", p.0));
            }
        }
        for v in blk.reads.iter().chain(blk.versions.iter().flat_map(|v| &v.body)) {
            let bad = matches!(cfg[*v].op, Op::Param(_) | Op::Removed) || v.is_reserved();
            if bad || !seen.insert(*v) {
                return Err(format!("bb{i}: bad or duplicate instr %{}", v.0));
            }
        }
    }

    for (i, blk) in cfg.blocks.iter().enumerate() {
        let b = BlockId(i as u32);
        if blk.origin.is_none() && blk.versions.last().is_none_or(|v| !v.guard.checks.is_empty()) {
            return Err(format!("bb{i}: nothing runs when every guard fails"));
        }
        // Reads read params and earlier reads (and only read); a Project
        // follows its LoadMethod or a sibling, as in a body.
        let params: HashSet<ValueId> = blk.params.iter().copied().collect();
        let mut reads: HashSet<ValueId> = HashSet::new();
        let mut projected = HashSet::new();
        for (k, r) in blk.reads.iter().enumerate() {
            if !super::guard::guard_read(cfg, *r) {
                return Err(format!("bb{i}: %{} can't be a guard's read", r.0));
            }
            if cfg[*r].args.iter().any(|a| !a.is_reserved() && !params.contains(a) && !reads.contains(a)) {
                return Err(format!("bb{i}: read %{} reads what isn't a param or an earlier read", r.0));
            }
            if let Op::Project(n) = cfg[*r].op {
                let p = cfg[*r].args[0];
                let prev = k.checked_sub(1).map(|j| blk.reads[j]);
                let follows = prev.is_some_and(|q| q == p || (matches!(cfg[q].op, Op::Project(_)) && cfg[q].args[0] == p));
                if !follows || !matches!(cfg[p].op, Op::LoadMethod(_)) || n >= 2 || !projected.insert((p, n)) {
                    return Err(format!("bb{i}: read %{} isn't a LoadMethod's projection right after it", r.0));
                }
            }
            reads.insert(*r);
        }
        if !blk.reads.is_empty() && blk.origin.is_none() {
            return Err(format!("bb{i}: a block with reads needs an origin"));
        }
        for v in &blk.versions {
            for (x, c) in &v.guard.checks {
                let ok = match x {
                    None => c.global(),
                    Some(x) => !c.global() && (params.contains(x) || reads.contains(x)),
                };
                if !ok {
                    return Err(format!("bb{i}: a guard check is on what isn't a param or a read (or a global one on a value)"));
                }
            }
            if v.inputs.iter().any(|x| !reads.contains(x)) {
                return Err(format!("bb{i}: a body takes what isn't a read"));
            }
            if let Some(x) = v.body.iter().find(|x| super::inline::is_check(&cfg[**x].op)) {
                return Err(format!("bb{i}: %{} checks in a version's body", x.0));
            }
        }
        let n = blk.n_outputs();
        if blk.versions.iter().any(|v| v.outputs.len() != n) {
            return Err(format!("bb{i}: versions disagree on the number of outputs"));
        }
        if blk.term.outputs().iter().any(|o| *o as usize >= n) {
            return Err(format!("bb{i}: terminator reads a missing output"));
        }
        for e in blk.term.exits() {
            match (e, &cfg.boundary) {
                (Exit::Block(e), _) => {
                    let want = cfg.explicit_params(e.target).len();
                    if e.args.len() != want {
                        return Err(format!("bb{i} -> bb{}: {} args for {} params", e.target.0, e.args.len(), want));
                    }
                }
                (Exit::Return(_), None) => {}
                (Exit::Leave(args), Some(bd)) if args.len() == bd.n_outputs => {}
                _ => return Err(format!("bb{i}: return or leave doesn't match the Cfg's boundary")),
            }
        }
        let mut used = vec![false; blk.handlers.len()];
        for seg in blk.versions.iter().flat_map(|v| &v.unwind) {
            match &seg.handler {
                Handler::Block(k, args) => {
                    let Some(h) = blk.handlers.get(*k) else {
                        return Err(format!("bb{i}: unwind to missing handler {k}"));
                    };
                    used[*k] = true;
                    let want = cfg.explicit_params(*h).len();
                    if args.len() != want {
                        return Err(format!("bb{i} unwind -> bb{}: {} args for {} params", h.0, args.len(), want));
                    }
                }
                Handler::Outer(k, args) => {
                    let arity = cfg.boundary.as_ref().and_then(|bd| bd.handler_arity.get(*k));
                    if arity != Some(&args.len()) {
                        return Err(format!("bb{i}: unwind to outer handler {k} doesn't match the Cfg's boundary"));
                    }
                }
                Handler::Propagate => {}
            }
        }
        match blk.term.unwind() {
            Some(TermUnwind::Block(k, args)) => {
                let Some(h) = blk.handlers.get(*k) else {
                    return Err(format!("bb{i}: terminator unwinds to missing handler {k}"));
                };
                used[*k] = true;
                let want = cfg.explicit_params(*h).len();
                if args.len() != want {
                    return Err(format!("bb{i} terminator unwind -> bb{}: {} args for {} params", h.0, args.len(), want));
                }
            }
            Some(TermUnwind::Propagate) | None => {}
        }
        // The origin may unwind to handlers no version does.
        if blk.origin.is_none() && used.contains(&false) {
            return Err(format!("bb{i}: a handler no segment unwinds to"));
        }
        if let Some(origin) = &blk.origin {
            check_origin(cfg, b, origin)?;
        }
        for ver in &blk.versions {
            check_segments(cfg, b, ver)?;
            check_locality(cfg, b, ver)?;
        }
    }
    Ok(())
}

fn check_origin(cfg: &Cfg, b: BlockId, origin: &Cfg) -> Result<(), String> {
    let blk = &cfg[b];
    verify(origin).map_err(|e| format!("bb{} origin: {e}", b.0))?;
    let kinds = |c: &Cfg, ps: &[ValueId]| -> Vec<Option<ParamKind>> { ps.iter().map(|p| c.param_kind(*p)).collect() };
    if kinds(origin, &origin[ENTRY].params) != kinds(cfg, &blk.params) {
        return Err(format!("bb{}: origin's ENTRY params don't match the block's", b.0));
    }
    let want = Boundary {
        n_outputs: blk.n_outputs(),
        handler_arity: blk.handlers.iter().map(|h| cfg.explicit_params(*h).len()).collect(),
    };
    if origin.boundary.as_ref() != Some(&want) {
        return Err(format!("bb{}: origin's boundary doesn't match the block", b.0));
    }
    Ok(())
}

fn check_segments(cfg: &Cfg, b: BlockId, ver: &Version) -> Result<(), String> {
    for (k, seg) in ver.unwind.iter().enumerate() {
        if seg.from >= ver.body.len() || (k > 0 && seg.from <= ver.unwind[k - 1].from) {
            return Err(format!("bb{}: unwind segments out of order", b.0));
        }
        if k == 0 && seg.handler == Handler::Propagate {
            return Err(format!("bb{}: first unwind segment has no handler", b.0));
        }
        if k > 0 && seg.handler == ver.unwind[k - 1].handler {
            return Err(format!("bb{}: adjacent identical unwind segments", b.0));
        }
        let to = ver.unwind.get(k + 1).map_or(ver.body.len(), |s| s.from);
        if !ver.body[seg.from..to].iter().any(|v| cfg[*v].op.can_raise()) {
            return Err(format!("bb{}: unwind segment {k} covers no raising op", b.0));
        }
    }
    Ok(())
}

// A multi-output op hands each output to its Project when it runs (the
// executor moves the owned reference), so a Project sits right after its
// producer or a sibling Project, and each output index has at most one.
fn check_project(cfg: &Cfg, ver: &Version, k: usize, i: u32, seen: &mut HashSet<(ValueId, u32)>) -> Result<(), String> {
    let p = cfg[ver.body[k]].args[0];
    let prev = k.checked_sub(1).map(|j| ver.body[j]);
    let adjacent = prev.is_some_and(|q| q == p || (matches!(cfg[q].op, Op::Project(_)) && cfg[q].args[0] == p));
    if !adjacent {
        return Err(format!("doesn't follow its producer %{}", p.0));
    }
    let n = match &cfg[p].op {
        Op::LoadMethod(_) | Op::IterNext | Op::Send => 2,
        // As many as the overload returns, checked when it runs.
        Op::Aten { .. } => usize::MAX,
        Op::Autograd(t) => cfg.trainings.get(*t as usize).map_or(0, |t| t.outputs.len()),
        Op::Region(r) => cfg.regions.get(*r as usize).map_or(0, |r| r.outputs),
        Op::Bytecode { opcode, arg } => opcodes::signature(*opcode, *arg, false).map_or(0, |s| {
            let all_new = s.outputs.iter().all(|o| *o == opcodes::Out::New);
            if all_new && s.outputs.len() > 1 { s.outputs.len() } else { 0 }
        }),
        _ => 0,
    };
    if i as usize >= n {
        return Err(format!("projects output {i} of %{}, which has {n} projectable outputs", p.0));
    }
    if !seen.insert((p, i)) {
        return Err(format!("a second projection of output {i} of %{}", p.0));
    }
    Ok(())
}

// Values are local to a version: every use is a reserved value, a param of
// the block, or an earlier instr of the same version. Segment args must be
// defined before the segment starts.
fn check_locality(cfg: &Cfg, b: BlockId, ver: &Version) -> Result<(), String> {
    let mut defined: HashSet<ValueId> = cfg[b].params.iter().chain(&ver.inputs).copied().collect();
    let ok = |defined: &HashSet<ValueId>, v: ValueId| v.is_reserved() || defined.contains(&v);
    let mut projected = HashSet::new();
    let mut seg = 0;
    for (k, v) in ver.body.iter().enumerate() {
        while seg < ver.unwind.len() && ver.unwind[seg].from == k {
            if let Handler::Block(_, args) | Handler::Outer(_, args) = &ver.unwind[seg].handler {
                if let Some(a) = args.iter().find(|a| !ok(&defined, **a)) {
                    return Err(format!("bb{} unwind: %{} isn't defined before the segment", b.0, a.0));
                }
            }
            seg += 1;
        }
        if let Some(a) = cfg[*v].args.iter().find(|a| !ok(&defined, **a)) {
            return Err(format!("bb{} instr %{}: %{} isn't defined earlier in the block", b.0, v.0, a.0));
        }
        if let Op::Project(i) = cfg[*v].op {
            check_project(cfg, ver, k, i, &mut projected).map_err(|e| format!("bb{} instr %{}: {e}", b.0, v.0))?;
        }
        defined.insert(*v);
    }
    if let Some(o) = ver.outputs.iter().find(|o| !ok(&defined, **o)) {
        return Err(format!("bb{}: output %{} isn't defined in the block", b.0, o.0));
    }
    Ok(())
}
