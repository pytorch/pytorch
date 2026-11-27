// What a recorded run KNOWS of the values that are no tensors, and the
// reasons a finished recording is not the body (record.rs asks; nothing
// else does). A value is known only POSITIVELY: by where it was produced.
//
// What it is given
//   - for each instruction of the recorded run that only moves a value (a
//     constant, a load or a store of a local), what it is (plain); for any
//     other, whether its result is known and, if not, what made it (set);
//   - at a conditional jump under the body, the value it tested (jump asks
//     lit / maker);
//   - whatever makes the recording less than the body (Reasons::add).
// What it guarantees
//   - lit() is true only for a value that is a constant of the code object,
//     a copy of one through locals the run itself stored, or one record.rs
//     set as a fact that the list checks (the key of an input or a result);
//     a local the run did not store, a parameter of a frame under the body,
//     a default, a value read from any object: NOT known;
//   - NEUTRAL names every kind of instruction that a recorded run may run
//     with nothing said in the list: the kinds that only move values inside
//     the run's own frame or make a new exact container of their operands.
//     A kind that is not named is a reason;
//   - Reasons keeps every distinct reason in the order they were met, with
//     counts; the first is the one a status names.
// What it may not assume
//   - that a value of an immutable class is the same on a later run.

use std::collections::{HashMap, HashSet};

use crate::cfg::ir::Op;
use crate::cfg::opcodes::*;
use crate::exec::rules::{Enters, Read, Row};

// An instruction that only moves a value: the constant, the load of a local
// slot, the store of a value into a slot, the slot emptied.
pub enum Plain {
    Const,
    Load(u32),
    Store(u32, u32),
    Delete(u32),
}

impl Plain {
    pub fn of(op: &Op, args: &[crate::cfg::ir::ValueId]) -> Option<Plain> {
        Some(match op {
            Op::Const(_) => Plain::Const,
            Op::LoadLocal { slot } | Op::LoadLocalChecked { slot } => Plain::Load(*slot),
            Op::StoreLocal { slot } => Plain::Store(*slot, args.first()?.0),
            Op::DeleteLocal { slot } => Plain::Delete(*slot),
            _ => return None,
        })
    }
}

// The kinds a recorded run may run with nothing said in the list, each with
// why no later run could see it do anything else. `row`: the instruction's
// row of the rule table, which must agree: it reads nothing but slots of
// the frame, constants of the code, which object an operand is and the
// interpreter's pending work, and it enters no frame it names.
pub fn neutral(op: &Op, row: &Row) -> Option<&'static str> {
    let reads = row.reads.iter().all(|r| matches!(r, Read::Slot | Read::Constant | Read::Identity | Read::Pending));
    if !reads || row.enters == Enters::Named {
        return None;
    }
    Some(match op {
        Op::Const(_) => "a constant of the code object, and the code is what a list is found by",
        Op::LoadLocal { .. } | Op::LoadLocalChecked { .. } | Op::StoreLocal { .. } | Op::DeleteLocal { .. } => "a slot of the run's own frame: the value keeps the origin it has, and a value the store lets go is released as eager releases it",
        Op::Project(_) => "it names one result of the instruction before it",
        Op::IsNone => "which object the operand is: the answer is known exactly when the operand is",
        Op::MakeCell { .. } => "it wraps a slot of the run's own frame in a new cell",
        // A frame of that work is a frame no instruction called: a stop.
        Op::Resume | Op::EvalBreaker => "the interpreter's pending work, none of which is the body's",
        Op::Bytecode { opcode: NOP | POP_TOP | PUSH_NULL | COPY | SWAP | END_FOR | CACHE | KW_NAMES | EXTENDED_ARG, .. } => "it moves values on the frame's own stack",
        Op::Bytecode { opcode: IS_OP, .. } => "whether two operands are one object: known exactly when both are",
        _ => return None,
    })
}

#[derive(Default)]
pub struct Known {
    // (activation, value): a value that is known.
    lit: HashSet<(u32, u32)>,
    // (activation, local slot): the slot holds a known value.
    slots: HashSet<(u32, u32)>,
    // (activation, value) -> what made a value that is not known.
    makers: HashMap<(u32, u32), String>,
}

impl Known {
    pub fn plain(&mut self, depth: u32, v: u32, op: Plain) {
        match op {
            Plain::Const => drop(self.lit.insert((depth, v))),
            Plain::Load(slot) => {
                if self.slots.contains(&(depth, slot)) {
                    self.lit.insert((depth, v));
                } else {
                    self.makers.insert((depth, v), "a local the run did not store a known value in (an input, a parameter)".into());
                }
            }
            Plain::Store(slot, x) => {
                if self.lit.contains(&(depth, x)) {
                    self.slots.insert((depth, slot));
                } else {
                    self.slots.remove(&(depth, slot));
                }
            }
            Plain::Delete(slot) => drop(self.slots.remove(&(depth, slot))),
        }
    }

    pub fn lit(&self, at: (u32, u32)) -> bool {
        self.lit.contains(&at)
    }

    // The value `at` is known (`lit`), or made by `maker`.
    pub fn set(&mut self, at: (u32, u32), lit: bool, maker: impl FnOnce() -> String) {
        if lit {
            self.lit.insert(at);
        } else {
            self.lit.remove(&at);
            self.makers.insert(at, maker());
        }
    }

    pub fn maker(&self, at: (u32, u32)) -> &str {
        self.makers.get(&at).map_or("a value of no recorded origin", |s| s.as_str())
    }

    // An activation is over.
    pub fn left(&mut self, depth: u32) {
        self.lit.retain(|(d, _)| *d != depth);
        self.slots.retain(|(d, _)| *d != depth);
        self.makers.retain(|(d, _), _| *d != depth);
    }
}

// Why a recording that reached its end is not the body, in the order met.
#[derive(Default)]
pub struct Reasons(pub Vec<(String, u64)>);

impl Reasons {
    pub fn add(&mut self, why: String) {
        match self.0.iter_mut().find(|(w, _)| *w == why) {
            Some((_, n)) => *n += 1,
            None => self.0.push((why, 1)),
        }
    }
}
