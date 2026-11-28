// The "not movable" rows of torch.vm.regions: where a region ends, and why.
//
// A region ends in front of an instruction the optimizer does not move:
// one its verdict refused (a call of unknown C code, a store, an operator
// that changes its operand in place, a read of a tensor's value) and that
// no other rule takes. Whoever reads the report decides from these rows
// what to rewrite, so a row has to be true of the plan that runs: a
// verdict is asked several times while a code is observed, and what one
// round refused a later round, or another rule of the same round, may take
// into a region.
//
// What it is given
//   - by the optimizer (mod.rs's optimize_now), each time it plans a code:
//     the Compiled, the Cfg the plan starts from with what was observed on
//     it, the blocks planned, the instructions the plan moves, and the
//     reasons the verdicts of this plan gave, by (Prepared, instruction);
//   - by the report (mod.rs's _cfg_impure): the Compileds that are there.
// What it guarantees
//   - all(): the rows of the last plan of every Compiled that is there, as
//     (name, line, reason), each once per Compiled. A row of an earlier
//     plan is gone with that plan, and the rows of a Compiled that was
//     dropped (a call site's clone goes with its caller's plan) with it.
// What it may not assume
//   - that a name is one Compiled: the clones of a code at one depth share
//     it, so rows are kept by the Compiled's address.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use super::{Compiled, Prepared};
use crate::cfg::ir::{BlockId, ENTRY, ValueId};

const MAX_ROWS: usize = 4096;

// (Compiled, name, line, reason).
static ROWS: Mutex<Vec<(usize, String, i32, String)>> = Mutex::new(Vec::new());

// `c` was planned from `prep`'s Cfg: of `blocks`, the plan moves `movable`,
// and `refused` has the reasons of its verdicts. A call the plan leaves
// where it is gets its row wherever it is, another instruction in a block a
// frame was seen in. `whole`: the plan is of the whole code (a plan of one
// block adds to the rows of the others).
pub(super) fn planned(c: &Compiled, prep: &Prepared, blocks: &[BlockId], movable: &HashSet<ValueId>, refused: &HashMap<(usize, u32), String>, whole: bool) {
    let (cfg, types) = (&prep.cfg, unsafe { &*prep.types.get() });
    let ran = |v: &ValueId| types.get(v.idx()).is_some_and(|t| t.ty.is_some());
    let (compiled, name) = (c as *const Compiled as usize, format!("{}@{}", c.qualname, c.depth));
    let mut rows = ROWS.lock().unwrap();
    if whole {
        rows.retain(|(at, ..)| *at != compiled);
    }
    for b in blocks.iter().filter(|b| b.idx() < cfg.blocks.len()) {
        let body = cfg[*b].versions.first().map_or(&[][..], |ver| &ver.body[..]);
        let entered = *b == ENTRY || cfg[*b].params.iter().any(ran) || body.iter().any(ran);
        for v in body.iter().filter(|v| !movable.contains(v) && (entered || super::is_call(&cfg[**v].op))) {
            let Some(why) = refused.get(&(prep as *const Prepared as usize, v.0)) else { continue };
            let line = c.line(cfg[*v].origin);
            if rows.len() < MAX_ROWS && !rows.iter().any(|(at, _, l, w)| *at == compiled && *l == line && w == why) {
                rows.push((compiled, name.clone(), line, why.clone()));
            }
        }
    }
}

// The rows of the Compileds at `live`.
pub(super) fn all(live: &HashSet<usize>) -> Vec<(String, i32, String)> {
    let mut rows = ROWS.lock().unwrap();
    rows.retain(|(c, ..)| live.contains(c));
    rows.iter().map(|(_, name, line, why)| (name.clone(), *line, why.clone())).collect()
}
