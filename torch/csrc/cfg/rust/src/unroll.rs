//! Unrolling of `for i in range(k)` loops with a small constant `k`.
//!
//! The loop is replaced by `k` copies of its body, each with the loop variable
//! as a constant, so that every iteration is straight-line code with blocks of
//! its own to specialize. Like inlining, the assumption (that `range` is the
//! builtin) is a branch whose other side keeps the loop as it was.
//!
//! ```text
//!     pre:    %it = get_iter range(3); jump header
//!     header: %i = iter_next %it; br_exhausted %i exit body
//!     body:   ...; jump header
//!  =>
//!     pre:    ...; jump guard
//!     guard:  br (range is builtins.range) body.0 header
//!     body.0: %i.0 = const 0; ...; jump body.1
//!     body.1: %i.1 = const 1; ...; jump body.2
//!     body.2: %i.2 = const 2; ...; jump tail
//!     tail:   jump exit
//!     exit:   phis of what the loop carried out, from header and tail
//! ```
//!
//! Loops with `break` or `return` in the body are left alone for now: their
//! only exit has to be the exhausted iterator.

use crate::frontend::{rewrite_term_uses, rewrite_uses};
use crate::ir::*;
use crate::pyffi::PyObj;
use pyo3::ffi;
use std::collections::{HashMap, HashSet};

const MAX_TRIP_COUNT: i64 = 8;
const MAX_UNROLLED_INSTS: usize = 600;

struct Loop {
    header: usize,
    /// The block that creates the iterator and jumps to the header.
    pre: usize,
    exit: usize,
    /// Blocks of the loop, header first.
    blocks: Vec<usize>,
    latches: Vec<usize>,
    trip_count: i64,
    /// The value `range` was loaded into.
    range_value: V,
}

fn defs(func: &Function) -> HashMap<V, (usize, usize)> {
    let mut out = HashMap::new();
    for (b, block) in func.blocks.iter().enumerate() {
        for (i, inst) in block.insts.iter().enumerate() {
            if inst.dst != NONE {
                out.insert(inst.dst, (b, i));
            }
        }
    }
    out
}

fn for_each_use(block: &Block, mut f: impl FnMut(V)) {
    for phi in &block.phis {
        phi.srcs.iter().for_each(|v| f(*v));
    }
    for inst in &block.insts {
        inst.op.for_each_use(&mut f);
    }
    block.term.for_each_use(&mut f);
}

/// The trip count if `v` is `range(k)` called on the builtin with a small
/// constant `k`, and the value `range` was loaded into.
unsafe fn constant_range(func: &Function, defs: &HashMap<V, (usize, usize)>, v: V) -> Option<(i64, V)> {
    let op = |v: V| defs.get(&v).map(|(b, i)| &func.blocks[*b].insts[*i].op);
    let Op::Call { callee, args, kwnames } = op(v)? else {
        return None;
    };
    let (Op::LoadGlobal(name), [count], true) = (op(*callee)?, args.as_slice(), *kwnames == NONE) else {
        return None;
    };
    let Op::Const(count) = op(*count)? else {
        return None;
    };
    let name = func.consts[*name as usize].as_ptr();
    let range = std::ptr::addr_of_mut!(ffi::PyRange_Type) as *mut ffi::PyObject;
    let shadowed = !ffi::PyDict_GetItemWithError(func.globals.as_ptr(), name).is_null();
    if shadowed || ffi::PyDict_GetItemWithError(func.builtins.as_ptr(), name) != range {
        ffi::PyErr_Clear();
        return None;
    }
    let count = func.consts[*count as usize].as_ptr();
    if ffi::PyLong_CheckExact(count) == 0 {
        return None;
    }
    let k = ffi::PyLong_AsLongLong(count);
    if !(0..=MAX_TRIP_COUNT).contains(&k) {
        ffi::PyErr_Clear();
        return None;
    }
    Some((k, *callee))
}

fn find_loop(func: &Function, done: &HashSet<usize>) -> Option<Loop> {
    let defs = defs(func);
    'headers: for (h, block) in func.blocks.iter().enumerate() {
        if done.contains(&h) {
            continue;
        }
        let ([inst], Term::BrNull { v, null, nonnull }) = (block.insts.as_slice(), &block.term) else {
            continue;
        };
        let Op::IterNext(it) = &inst.op else {
            continue;
        };
        if inst.dst != *v {
            continue;
        }
        let Some(&(pre, at)) = defs.get(it) else {
            continue;
        };
        let Op::GetIter(iterable) = &func.blocks[pre].insts[at].op else {
            continue;
        };
        let Some((trip_count, range_value)) = (unsafe { constant_range(func, &defs, *iterable) }) else {
            continue;
        };
        let exit = *null as usize;
        if !matches!(func.blocks[pre].term, Term::Jump(t) if t as usize == h)
            || func.blocks[exit].preds != [h as B]
        {
            continue;
        }
        // The natural loop: what reaches a latch without passing the header.
        let latches: Vec<usize> =
            block.preds.iter().map(|p| *p as usize).filter(|p| *p != pre).collect();
        let mut blocks = vec![h];
        let mut stack = latches.clone();
        while let Some(b) = stack.pop() {
            if !blocks.contains(&b) {
                blocks.push(b);
                stack.extend(func.blocks[b].preds.iter().map(|p| *p as usize));
            }
        }
        if latches.is_empty() || blocks.contains(&pre) || !blocks.contains(&(*nonnull as usize)) {
            continue;
        }
        let in_loop: HashSet<usize> = blocks.iter().copied().collect();
        // One way in, one way out, and the iterator is only stepped here.
        let mut size = 0;
        let mut defined: HashSet<V> = HashSet::new();
        for &b in &blocks {
            let body = &func.blocks[b];
            size += body.insts.len();
            let inside = |x: &B| in_loop.contains(&(*x as usize));
            if b != h && !(body.preds.iter().all(inside) && body.term.successors().iter().all(inside)) {
                continue 'headers;
            }
            defined.extend(body.insts.iter().map(|inst| inst.dst));
            if b != h {
                defined.extend(body.phis.iter().map(|phi| phi.dst));
            }
        }
        if size * trip_count as usize > MAX_UNROLLED_INSTS {
            continue;
        }
        let mut iterator_uses = 0;
        let mut escapes = false;
        for (b, other) in func.blocks.iter().enumerate() {
            for_each_use(other, |used| {
                iterator_uses += (used == *it) as usize;
                // Past the loop, only what the header's phis carry is visible.
                escapes |= !in_loop.contains(&b) && defined.contains(&used);
            });
        }
        if iterator_uses != 1 || escapes {
            continue;
        }
        return Some(Loop { header: h, pre, exit, blocks, latches, trip_count, range_value });
    }
    None
}

fn new_value(func: &mut Function, like: V) -> V {
    let name = func.value_names[like as usize].clone();
    func.value_names.push(name);
    func.nvalues += 1;
    func.nvalues - 1
}

fn new_const(func: &mut Function, obj: PyObj) -> u32 {
    func.consts.push(obj);
    func.consts.len() as u32 - 1
}

fn unroll(func: &mut Function, lp: &Loop) {
    let h = lp.header;
    let k = lp.trip_count as usize;
    let n = lp.blocks.len();
    let guard = func.blocks.len();
    let copy = |j: usize, pos: usize| (guard + 1 + j * n + pos) as B;
    let tail = guard + 1 + k * n;
    let position: HashMap<usize, usize> = lp.blocks.iter().enumerate().map(|(pos, b)| (*b, pos)).collect();
    let header_phis = func.blocks[h].phis.clone();
    let header_preds: Vec<usize> = func.blocks[h].preds.iter().map(|p| *p as usize).collect();
    let pre_slot = header_preds.iter().position(|p| *p == lp.pre).unwrap();
    let latch_slots: Vec<usize> = (0..header_preds.len()).filter(|s| *s != pre_slot).collect();
    let loop_var = func.blocks[h].insts[0].clone();
    let bc = loop_var.bc;

    // The guard: `range` still is the builtin.
    let range_const = new_const(func, unsafe {
        PyObj::from_borrowed(std::ptr::addr_of_mut!(ffi::PyRange_Type) as *mut ffi::PyObject)
    });
    let range = new_value(func, UNDEF);
    let holds = new_value(func, UNDEF);
    let entry = if k == 0 { tail as B } else { copy(0, 0) };
    let guard_block = Block {
        phis: Vec::new(),
        insts: vec![
            Inst { dst: range, op: Op::Const(range_const), bc, kills: Vec::new() },
            Inst { dst: holds, op: Op::Is(lp.range_value, range, false), bc, kills: Vec::new() },
        ],
        term: Term::Br { cond: holds, then_: entry, else_: h as B },
        preds: vec![lp.pre as B],
        bc_start: bc,
    };
    func.blocks[lp.pre].term = Term::Jump(guard as B);
    func.blocks[h].preds[pre_slot] = guard as B;
    func.blocks.push(guard_block);

    // What the header's phis hold on entry to copy j, given the values of
    // copy j - 1 (`prev`). Copy k is the tail.
    let mut prev: HashMap<V, V> = HashMap::new();
    for j in 0..=k {
        let mut values: HashMap<V, V> = HashMap::new();
        let mut phis = Vec::new();
        let preds: Vec<B> = if j == 0 {
            vec![guard as B]
        } else {
            latch_slots.iter().map(|s| copy(j - 1, position[&header_preds[*s]])).collect()
        };
        for phi in &header_phis {
            let value = if j == 0 {
                phi.srcs[pre_slot]
            } else if let [slot] = latch_slots.as_slice() {
                let src = phi.srcs[*slot];
                prev.get(&src).copied().unwrap_or(src)
            } else {
                let dst = new_value(func, phi.dst);
                let srcs = latch_slots
                    .iter()
                    .map(|s| prev.get(&phi.srcs[*s]).copied().unwrap_or(phi.srcs[*s]))
                    .collect();
                phis.push(Phi { dst, srcs });
                dst
            };
            values.insert(phi.dst, value);
        }
        if j == k {
            // The tail stands for the header finding the iterator exhausted.
            let exit = lp.exit;
            func.blocks.push(Block { phis, insts: Vec::new(), term: Term::Jump(exit as B), preds, bc_start: bc });
            func.blocks[exit].preds.push(tail as B);
            let mut renamed: HashMap<V, V> = HashMap::new();
            let in_loop: HashSet<usize> = lp.blocks.iter().copied().collect();
            for phi in &header_phis {
                let merged = new_value(func, phi.dst);
                func.blocks[exit].phis.push(Phi { dst: merged, srcs: vec![phi.dst, values[&phi.dst]] });
                renamed.insert(phi.dst, merged);
            }
            for (b, block) in func.blocks.iter_mut().enumerate().take(guard) {
                if in_loop.contains(&b) {
                    continue;
                }
                let rename = |v: V| renamed.get(&v).copied().unwrap_or(v);
                // The exit's new phis read the loop's own values.
                if b != exit {
                    for phi in block.phis.iter_mut() {
                        phi.srcs.iter_mut().for_each(|v| *v = rename(*v));
                    }
                }
                for inst in block.insts.iter_mut() {
                    let _ = rewrite_uses(&mut inst.op, &mut |v| Ok(rename(v)));
                }
                let _ = rewrite_term_uses(&mut block.term, &mut |v| Ok(rename(v)));
            }
            break;
        }
        // Copy j of the loop.
        let int = new_const(func, unsafe { PyObj::from_owned(ffi::PyLong_FromLongLong(j as i64)) });
        for &b in &lp.blocks {
            let block = &func.blocks[b];
            let mut dsts: Vec<V> = block.insts.iter().map(|inst| inst.dst).collect();
            if b != h {
                dsts.extend(block.phis.iter().map(|phi| phi.dst));
            }
            for dst in dsts {
                if dst != NONE {
                    let fresh = new_value(func, dst);
                    values.insert(dst, fresh);
                }
            }
        }
        let map = |v: V| values.get(&v).copied().unwrap_or(v);
        let next_header = if j + 1 == k { tail as B } else { copy(j + 1, 0) };
        let target = |t: B| if t as usize == h { next_header } else { copy(j, position[&(t as usize)]) };
        for (pos, &b) in lp.blocks.iter().enumerate() {
            let block = &func.blocks[b];
            let cloned = if b == h {
                Block {
                    phis: std::mem::take(&mut phis),
                    insts: vec![Inst { dst: map(loop_var.dst), op: Op::Const(int), bc, kills: Vec::new() }],
                    term: Term::Jump(match &block.term {
                        Term::BrNull { nonnull, .. } => copy(j, position[&(*nonnull as usize)]),
                        _ => unreachable!(),
                    }),
                    preds: preds.clone(),
                    bc_start: block.bc_start,
                }
            } else {
                let mut insts = block.insts.clone();
                for inst in insts.iter_mut() {
                    inst.dst = map(inst.dst);
                    let _ = rewrite_uses(&mut inst.op, &mut |v| Ok(map(v)));
                }
                let mut term = block.term.clone();
                let _ = rewrite_term_uses(&mut term, &mut |v| Ok(map(v)));
                let term = match term {
                    Term::Jump(t) => Term::Jump(target(t)),
                    Term::Br { cond, then_, else_ } => Term::Br { cond, then_: target(then_), else_: target(else_) },
                    Term::BrNull { v, null, nonnull } => {
                        Term::BrNull { v, null: target(null), nonnull: target(nonnull) }
                    }
                    other => other,
                };
                Block {
                    phis: block
                        .phis
                        .iter()
                        .map(|phi| Phi { dst: map(phi.dst), srcs: phi.srcs.iter().map(|v| map(*v)).collect() })
                        .collect(),
                    insts,
                    term,
                    preds: block.preds.iter().map(|p| copy(j, position[&(*p as usize)])).collect(),
                    bc_start: block.bc_start,
                }
            };
            debug_assert_eq!(func.blocks.len(), copy(j, pos) as usize);
            func.blocks.push(cloned);
        }
        prev = values;
    }
}

/// Unrolls every loop that qualifies, and returns how many were.
pub fn unroll_loops(func: &mut Function) -> usize {
    // Headers that were unrolled stay behind as the fallback of their guard.
    let mut done: HashSet<usize> = HashSet::new();
    let mut unrolled = 0;
    while let Some(lp) = find_loop(func, &done) {
        done.insert(lp.header);
        unroll(func, &lp);
        unrolled += 1;
    }
    unrolled
}
