// Guards: what an optimized version checks on entry, before its body runs.
//
// Elision and specialization copy code into a version along the path it was
// seen to take, with an Expect of a Check wherever that path depends
// on something: a branch's condition, the callee of an elided call. Those
// checks must all hold before the body runs, so that a failed one is a miss
// with nothing to undo (the block's origin runs instead), and so that the
// body can be optimized as a whole. lift_guard moves them out of the body:
// what they check becomes a read of the block (Block::reads: an attribute of
// a param, a global, an item of what an attribute holds, ...), and each
// Expect its check on that read in the guard. Only what checks need moves,
// what the block's compute (aten ops, pinned calls) takes, its graph's
// inputs, and what it only passes on: reads feeding Python code stay in the
// body where they were. A read in the guard
// that the body uses becomes an input of the body. Reads are shared: the
// same attribute read twice is read once.

use std::collections::{HashMap, HashSet};

use super::inline::{is_check, is_read};
use super::ir::*;
use super::opcodes;

// Lifts the checks of block p's only version into its guard, with the reads
// they need: in body order, each read whose args are params or reads already
// lifted (and which `stale` doesn't keep in place) can be one of
// Block::reads, and each check of a param or such a read becomes a check of
// the guard; then the reads that only feed Python code go back. The body keeps the rest,
// and takes the reads it uses (or a handler it raises into takes) as inputs;
// `handler` gives where each body instr raises to. False if a check stays in
// the body (it checks something the body computes). `stale` sees every body
// instr in order (see specialize.rs).
pub fn lift_guard(cfg: &mut Cfg, p: BlockId, mut stale: impl FnMut(&Cfg, ValueId) -> bool, handler: impl Fn(ValueId) -> Handler) -> bool {
    let body = std::mem::take(&mut cfg[p].versions[0].body);
    let params: HashSet<ValueId> = cfg[p].params.iter().copied().collect();
    let mut reads = std::mem::take(&mut cfg[p].reads);
    let mut lifted: HashSet<ValueId> = reads.iter().copied().collect();
    let mut seen: HashMap<(String, Vec<ValueId>), ValueId> = reads.iter().map(|v| ((format!("{:?}", cfg[*v].op), cfg[*v].args.clone()), *v)).collect();
    let mut same: HashMap<ValueId, ValueId> = HashMap::new();
    let mut on = vec![];
    let mut rest = vec![];
    let mut ok = true;
    let order = body.clone();
    // Each constant copied among the reads, and a constant it copies.
    let mut copy_of: HashMap<ValueId, ValueId> = HashMap::new();
    // Constants stay in the body; a read or check taking one gets its own
    // copy among the reads.
    let constant = |op: &Op| matches!(op, Op::Const(_) | Op::Lit(_) | Op::Pinned(_));
    for x in body {
        let args: Vec<ValueId> = cfg[x].args.iter().map(|a| *same.get(a).unwrap_or(a)).collect();
        cfg[x].args = args.clone();
        let kept = stale(cfg, x);
        let known = |a: &ValueId| a.is_reserved() || params.contains(a) || lifted.contains(a) || constant(&cfg[*a].op);
        let op = cfg[x].op.clone();
        let read = guard_read(cfg, x) && !constant(&op);
        if !(is_check(&op) || read) || kept || !args.iter().all(known) {
            ok &= !is_check(&op);
            rest.push(x);
            continue;
        }
        let mut args = args;
        for a in args.iter_mut() {
            if a.is_reserved() || params.contains(a) || lifted.contains(a) {
                continue;
            }
            let key = (format!("{:?}", cfg[*a].op), vec![]);
            *a = match seen.get(&key) {
                Some(c) => *c,
                None => {
                    let c = cfg.add(cfg[*a].op.clone(), vec![], cfg[*a].origin);
                    seen.insert(key, c);
                    lifted.insert(c);
                    reads.push(c);
                    copy_of.insert(c, *a);
                    c
                }
            };
        }
        cfg[x].args = args.clone();
        let check = match op {
            Op::Expect(c) => *c,
            op => {
                let key = (format!("{op:?}"), args);
                match seen.get(&key) {
                    Some(v) => {
                        same.insert(x, *v);
                    }
                    None => {
                        seen.insert(key, x);
                        lifted.insert(x);
                        reads.push(x);
                    }
                }
                continue;
            }
        };
        // A constant can't change.
        let c = (Some(args[0]), check);
        if !constant(&cfg[args[0]].op) && !on.contains(&c) {
            on.push(c);
        }
    }
    // The reads the checks need (their targets), those compute takes (an
    // aten op's or pinned call's args: the graph's inputs, as in Dynamo) and
    // those the block only passes on (outputs, handler args), and what those
    // read (with a LoadMethod's other output). The others, which feed Python
    // code, go back where they were.
    let compute_args = rest.iter().filter(|v| compute(cfg, **v)).flat_map(|v| cfg[*v].args.iter().copied());
    let outputs = cfg[p].versions[0].outputs.iter().map(|o| *same.get(o).unwrap_or(o));
    let handler_args = rest.iter().flat_map(|v| match handler(*v) {
        Handler::Block(_, args) | Handler::Outer(_, args) => args,
        Handler::Propagate => vec![],
    });
    let passed = outputs.chain(handler_args).map(|a| *same.get(&a).unwrap_or(&a)).collect::<Vec<_>>();
    let mut need: HashSet<ValueId> = on.iter().filter_map(|(x, _)| *x).chain(compute_args).chain(passed).filter(|x| lifted.contains(x)).collect();
    for r in reads.iter().rev() {
        let sibling = |r: &ValueId| matches!(cfg[*r].op, Op::Project(_)) && need.contains(&cfg[*r].args[0]);
        if need.contains(r) || sibling(r) {
            need.insert(*r);
            need.extend(cfg[*r].args.iter().copied().filter(|a| lifted.contains(a)));
        }
    }
    // A LoadMethod comes before its Projects: a second pass for siblings.
    for r in &reads {
        if matches!(cfg[*r].op, Op::Project(_)) && need.contains(&cfg[*r].args[0]) {
            need.insert(*r);
        }
    }
    let back: HashSet<ValueId> = reads.iter().copied().filter(|r| !need.contains(r) && !copy_of.contains_key(r)).collect();
    for r in &back {
        let args: Vec<ValueId> = cfg[*r].args.iter().map(|a| *copy_of.get(a).unwrap_or(a)).collect();
        cfg[*r].args = args;
    }
    reads.retain(|r| need.contains(r));
    let kept: HashSet<ValueId> = rest.iter().copied().collect();
    let rest: Vec<ValueId> = order.into_iter().filter(|x| kept.contains(x) || back.contains(x)).collect();
    let mut rest = rest;
    let ver = &mut cfg[p].versions[0];
    for o in ver.outputs.iter_mut() {
        *o = *same.get(o).unwrap_or(o);
    }
    // Constants only lifted reads used.
    let outputs = ver.outputs.clone();
    let used: HashSet<ValueId> = rest.iter().flat_map(|v| cfg[*v].args.iter()).chain(&outputs).copied().collect();
    rest.retain(|v| !constant(&cfg[*v].op) || used.contains(v));
    let ver = &mut cfg[p].versions[0];
    ver.body = rest;
    ver.guard.checks.extend(on);
    ver.set_unwind(|x| match handler(x) {
        Handler::Block(k, args) => Handler::Block(k, args.iter().map(|a| *same.get(a).unwrap_or(a)).collect()),
        Handler::Outer(k, args) => Handler::Outer(k, args.iter().map(|a| *same.get(a).unwrap_or(a)).collect()),
        h => h,
    });
    cfg.normalize_unwind(p);
    let ver = &cfg[p].versions[0];
    let handler_args = ver.unwind.iter().flat_map(|s| match &s.handler {
        Handler::Block(_, args) | Handler::Outer(_, args) => args.as_slice(),
        Handler::Propagate => &[],
    });
    let used: HashSet<ValueId> = ver.body.iter().flat_map(|v| cfg[*v].args.iter()).chain(&ver.outputs).chain(handler_args).copied().collect();
    let inputs: Vec<ValueId> = reads.iter().copied().filter(|r| used.contains(r)).collect();
    cfg[p].versions[0].inputs = inputs;
    cfg[p].reads = reads;
    ok
}

// Whether instr x is compute: an aten op, or a call of a pinned callee (one
// torch op, see cfg/optimize.rs).
fn compute(cfg: &Cfg, x: ValueId) -> bool {
    let ins = &cfg[x];
    match ins.op {
        Op::Aten { .. } => true,
        Op::Call { .. } => ins.args.len() >= 2 && ins.args[0] == NULL && matches!(cfg[ins.args[1]].op, Op::Pinned(_)),
        _ => false,
    }
}

// Whether instr x may be one of Block::reads: a read (see inline::is_read)
// but Resume (the body's first instr) and an empty dict (a new object each
// time: shared, it made two elided calls' **kwargs one dict).
pub fn guard_read(cfg: &Cfg, x: ValueId) -> bool {
    let op = &cfg[x].op;
    is_read(cfg, x) && !is_check(op) && *op != Op::Resume && !matches!(op, Op::Bytecode { opcode: opcodes::BUILD_MAP, .. })
}

// Whether instr x may change what a read after it reads: it stores to an
// attribute, a local or a container, or calls something that isn't pinned
// (a pinned callee runs one torch op, see cfg/optimize.rs). Aten ops,
// operators on scalars and the reads themselves don't.
pub fn writes(cfg: &Cfg, x: ValueId) -> bool {
    let ins = &cfg[x];
    match ins.op {
        Op::StoreAttr(_)
        | Op::StoreLocal { .. }
        | Op::DeleteLocal { .. }
        | Op::MakeCell { .. }
        | Op::CopyFreeVars(_)
        | Op::StoreDeref { .. }
        | Op::DeleteDeref { .. }
        | Op::Yield
        | Op::Send
        | Op::IterNext => true,
        Op::Call { .. } => !is_read(cfg, x) && !(ins.args.len() >= 2 && ins.args[0] == NULL && matches!(cfg[ins.args[1]].op, Op::Pinned(_))),
        Op::Bytecode { opcode, .. } => {
            !is_read(cfg, x) && !matches!(opcode, opcodes::UNARY_NEGATIVE | opcodes::UNARY_NOT | opcodes::UNARY_INVERT | opcodes::BUILD_LIST | opcodes::BUILD_MAP)
        }
        _ => false,
    }
}
