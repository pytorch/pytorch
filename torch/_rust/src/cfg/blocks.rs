// Block partition: leaders, reachability, predecessor checks and RPO
// numbering. Exception-range boundaries and raising ops do not split blocks;
// unwind segments (built by the interpreter) cover them.

use std::collections::BTreeSet;

use super::bytecode::Decoded;
use super::ir::{rpo_by, BlockId};
use super::opcodes as oc;
use super::CfgError;

pub struct PBlock {
    // Instructions [start, end).
    pub start: usize,
    pub end: usize,
    // Normal successors in the terminator's field order.
    pub succs: Vec<BlockId>,
    // (depth, lasti) for exception-handler blocks.
    pub handler: Option<(usize, bool)>,
}

pub struct Partition {
    pub blocks: Vec<PBlock>,
    // Handler block per exception-table entry (None if unreachable).
    pub entry_block: Vec<Option<BlockId>>,
}

fn ends_block(op: u8) -> bool {
    matches!(
        op,
        oc::JUMP_FORWARD
            | oc::JUMP_BACKWARD
            | oc::JUMP_BACKWARD_NO_INTERRUPT
            | oc::POP_JUMP_IF_FALSE
            | oc::POP_JUMP_IF_TRUE
            | oc::POP_JUMP_IF_NONE
            | oc::POP_JUMP_IF_NOT_NONE
            | oc::FOR_ITER
            | oc::SEND
            | oc::RETURN_VALUE
            | oc::RETURN_CONST
            | oc::RAISE_VARARGS
            | oc::RERAISE
    )
}

// Normal successors of the block ending at instruction `i`, in the field order
// of the terminator it lowers to.
fn insn_successors(dec: &Decoded, i: usize) -> Result<Vec<usize>, CfgError> {
    let insn = &dec.insns[i];
    let fall = i + 1;
    let tgt = dec.jump_target(i)?;
    let need_fall = || {
        if fall < dec.insns.len() {
            Ok(fall)
        } else {
            Err(CfgError::unsupported(insn.unit * 2, oc::name(insn.op), "falls off the end"))
        }
    };
    Ok(match insn.op {
        oc::JUMP_FORWARD | oc::JUMP_BACKWARD | oc::JUMP_BACKWARD_NO_INTERRUPT => vec![tgt.unwrap()],
        oc::POP_JUMP_IF_TRUE | oc::POP_JUMP_IF_NONE => vec![tgt.unwrap(), need_fall()?],
        oc::POP_JUMP_IF_FALSE | oc::POP_JUMP_IF_NOT_NONE | oc::FOR_ITER | oc::SEND => {
            vec![need_fall()?, tgt.unwrap()]
        }
        oc::RETURN_VALUE | oc::RETURN_CONST | oc::RAISE_VARARGS | oc::RERAISE => vec![],
        _ => vec![need_fall()?],
    })
}

pub fn partition(dec: &Decoded) -> Result<Partition, CfgError> {
    let n = dec.insns.len();
    if n == 0 {
        return Err(CfgError::unsupported(0, "", "empty code"));
    }

    // 1. Leaders.
    let mut leaders = BTreeSet::from([0usize]);
    for i in 0..n {
        if let Some(t) = dec.jump_target(i)? {
            leaders.insert(t);
        }
        if ends_block(dec.insns[i].op) && i + 1 < n {
            leaders.insert(i + 1);
        }
    }
    leaders.extend(dec.entry_target.iter().copied());
    let starts: Vec<usize> = leaders.into_iter().collect();
    let mut raw_of_insn = vec![usize::MAX; n];
    for (r, s) in starts.iter().enumerate() {
        raw_of_insn[*s] = r;
    }
    let nraw = starts.len();
    let range = |r: usize| (starts[r], starts.get(r + 1).copied().unwrap_or(n));

    // Handler (depth, lasti) per raw block; every entry for a target must agree.
    let mut handler_info: Vec<Option<(usize, bool)>> = vec![None; nraw];
    for (e, ent) in dec.entries.iter().enumerate() {
        let r = raw_of_insn[dec.entry_target[e]];
        let info = (ent.depth, ent.lasti);
        match handler_info[r] {
            Some(prev) if prev != info => {
                return Err(CfgError::unsupported(
                    ent.target * 2,
                    "",
                    "handler target with two different depth/lasti",
                ))
            }
            _ => handler_info[r] = Some(info),
        }
    }

    let mut normal: Vec<Vec<usize>> = Vec::with_capacity(nraw);
    let mut handlers: Vec<Vec<usize>> = Vec::with_capacity(nraw);
    for r in 0..nraw {
        let (s, e) = range(r);
        normal.push(insn_successors(dec, e - 1)?.into_iter().map(|t| raw_of_insn[t]).collect());
        let entries: BTreeSet<usize> = (s..e).filter_map(|i| dec.insn_entry[i]).collect();
        handlers.push(entries.into_iter().map(|en| raw_of_insn[dec.entry_target[en]]).collect());
    }

    // 2. Reachability over normal and handler edges.
    let mut reach = vec![false; nraw];
    let mut work = vec![0usize];
    reach[0] = true;
    while let Some(r) = work.pop() {
        for t in normal[r].iter().chain(&handlers[r]).copied() {
            if !reach[t] {
                reach[t] = true;
                work.push(t);
            }
        }
    }

    // 3. Predecessor checks over normal edges of reachable blocks.
    for r in (0..nraw).filter(|r| reach[*r]) {
        for t in &normal[r] {
            if *t == 0 {
                return Err(CfgError::unsupported(0, "", "jump to the entry"));
            }
            if handler_info[*t].is_some() {
                let (s, _) = range(*t);
                return Err(CfgError::unsupported(
                    dec.insns[s].unit * 2,
                    "",
                    "normal control flow into an exception handler",
                ));
            }
        }
    }

    // 4. RPO numbering.
    let succs = |x: usize| -> Vec<usize> { normal[x].iter().chain(&handlers[x]).copied().collect() };
    let order = rpo_by(nraw, 0, succs);
    let mut id = vec![usize::MAX; nraw];
    for (k, x) in order.iter().enumerate() {
        id[*x] = k;
    }

    let blocks = order
        .iter()
        .map(|&x| {
            let (start, end) = range(x);
            PBlock {
                start,
                end,
                succs: normal[x].iter().map(|t| BlockId(id[*t] as u32)).collect(),
                handler: handler_info[x],
            }
        })
        .collect();
    let entry_block = dec
        .entry_target
        .iter()
        .map(|t| {
            let r = raw_of_insn[*t];
            reach[r].then(|| BlockId(id[r] as u32))
        })
        .collect();
    Ok(Partition {
        blocks,
        entry_block,
    })
}
