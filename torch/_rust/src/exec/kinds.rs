// The kinds of operands a verdict saw, asked again while a body is traced.
//
// shown.rs judges what C code does to an operand by the kinds the operand
// was seen to have. A region's trace is one more run, and the first on
// which an operand may be of another kind: an iterator object where every
// call before had a list. What the instruction does to that one (take its
// items, change it in place) would be done once, by the trace, and never by
// the region.
//
// What it is given
//   - by covered.rs, with each verdict: the kinds the operands of an
//     instruction it covered were seen to have (seen, Shown::add);
//   - by covered.rs, before an instruction of a traced body runs: its op,
//     the objects of its operands, and whether it calls a builtin that
//     takes the items of what it is handed (Shown::unseen).
// What it guarantees
//   - unseen() names the class of the first operand that the instruction
//     takes the items of, or changes in place, and that is of no kind a
//     verdict saw in that position of that instruction. The instructions:
//     GET_ITER and FOR_ITER (what is iterated), UNPACK_SEQUENCE and
//     UNPACK_EX (what is unpacked), CONTAINS_OP (the container),
//     LIST_EXTEND, SET_UPDATE, DICT_UPDATE and DICT_MERGE (the source), an
//     in-place binary operator (the left operand), a call of a builtin
//     that takes the items of its arguments or makes an iterator over them
//     (each argument);
//   - an operand that shown.rs passes whatever was seen is never named: a
//     number, a string, None, a tuple, a frozenset, a tensor; an exact
//     list, set or dict where its items are taken (by a new iterator); and
//     the trace's own stand-in for a float (scalars.rs), which is that
//     float.
// What it may not assume
//   - that a kind a verdict saw was judged for the frame that runs now
//     (what is kept is per code, as in covered.rs);
//   - anything of the instructions it is not asked about: a read in C code
//     (an attribute, an item, a length, a comparison) of an object of a
//     class no call showed is taken to have no effect, and a method written
//     in Python is a frame, asked about as code.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Mutex;

use pyo3::ffi::PyObject;

use super::types::{self, Ty};
use super::{opcodes, scalars, Op, Prepared, ValueId};

// The kinds of every operand of one instruction, as one verdict found them
// (None: never computed).
pub type Seen = Vec<Option<Ty>>;

// How many verdicts' kinds are kept for one instruction.
const MAX_SEEN: usize = 8;

// BINARY_OP's in-place operators start at NB_INPLACE_ADD.
const INPLACE: u32 = 13;

// Which operands of an instruction with op `op` and `n` operands are taken
// from or changed in place; `taker`: it calls a builtin that takes the items
// of what it is handed, and `null`: with NULL for its first operand (a call
// of a function, not of a method).
fn taken(op: &Op, n: usize, taker: bool, null: bool) -> Range<usize> {
    match op {
        Op::GetIter | Op::IterNext | Op::Bytecode { opcode: opcodes::UNPACK_SEQUENCE | opcodes::UNPACK_EX, .. } => 0..1,
        Op::Binary(k) if *k >= INPLACE => 0..1,
        Op::Bytecode { opcode: opcodes::CONTAINS_OP, .. } => 1..2,
        Op::Bytecode { opcode: opcodes::LIST_EXTEND | opcodes::SET_UPDATE | opcodes::DICT_UPDATE | opcodes::DICT_MERGE, .. } => n.saturating_sub(1)..n,
        Op::Call { .. } if taker => (if null { 2 } else { 1 })..n,
        _ => 0..0,
    }
}

// The operand that an instruction with op `op` fills in place, if it is one
// of those that build a display or a call's arguments: the container below
// what it adds (DICT_MERGE's operands begin with the function, for its
// error).
pub fn filled(op: &Op) -> Option<usize> {
    match op {
        Op::Bytecode { opcode: opcodes::LIST_APPEND | opcodes::SET_ADD | opcodes::MAP_ADD | opcodes::LIST_EXTEND | opcodes::SET_UPDATE | opcodes::DICT_UPDATE, .. } => Some(0),
        Op::Bytecode { opcode: opcodes::DICT_MERGE, .. } => Some(1),
        _ => None,
    }
}

// Whether an instruction with op `op` may be one that is asked about, or is
// one whose operands covered.rs is to see for the container it fills.
pub fn asked(op: &Op) -> bool {
    !taken(op, 3, true, false).is_empty() || filled(op).is_some()
}

// What a verdict that covers instr v of `prep` saw of its operands, if v is
// asked about (`taker` as for taken).
pub fn seen(prep: &Prepared, v: ValueId, taker: bool) -> Option<Seen> {
    let ins = &prep.cfg[v];
    let types = unsafe { &*prep.types.get() };
    (!taken(&ins.op, ins.args.len(), taker, false).is_empty()).then(|| ins.args.iter().map(|a| types.get(a.idx()).and_then(|t| t.ty.clone())).collect())
}

// By code unit: what each verdict on the instruction there saw.
#[derive(Default)]
pub struct Shown(Mutex<HashMap<u32, Vec<Seen>>>);

impl Shown {
    pub fn add(&self, unit: u32, seen: Seen) {
        let mut all = self.0.lock().unwrap();
        let known = all.entry(unit).or_default();
        if known.len() < MAX_SEEN && !known.contains(&seen) {
            known.push(seen);
        }
    }

    // The class of the first operand among `objs` (borrowed, NULL allowed)
    // that the instruction at `unit` takes from or changes and that no
    // verdict saw there.
    pub unsafe fn unseen(&self, unit: u32, op: &Op, objs: &[*mut PyObject], taker: bool) -> Option<String> {
        let at = taken(op, objs.len(), taker, objs.first().is_some_and(|o| o.is_null()));
        let inplace = matches!(op, Op::Binary(_));
        let all = self.0.lock().unwrap();
        let known = all.get(&unit).map_or(&[][..], |k| &k[..]);
        for i in at {
            let obj = objs[i];
            if obj.is_null() || scalars::stand_in(obj) || !types::by_class(obj, inplace) {
                continue;
            }
            let saw = |s: &Seen| s.len() == objs.len() && s[i].as_ref().is_some_and(|t| t.saw(obj));
            if !known.iter().any(saw) {
                return Some(types::class_name(obj));
            }
        }
        None
    }
}
