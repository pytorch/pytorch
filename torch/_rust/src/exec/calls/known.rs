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
//     a copy of one through locals the run itself stored, one record.rs
//     set as a fact that the list checks (the key of an input or a result),
//     an input that the list was specialized on (input(): a param of the
//     block, or a local of the body's frame that the run had not stored
//     into when it loaded it; origins.rs keeps the value, and a later
//     arrival is asked for exactly it), or the result of an instruction of
//     one of the kinds of FOUND for which reads.rs holds the read that gave
//     it. A parameter of a frame under the body, a default, the content of
//     a cell, a value read from any object by any other kind: NOT known;
//   - FOUND, the kinds whose result IS what a recorded read found, each with
//     the statement that makes it so (Found::of) and the door of reads.rs
//     that is asked behind the instruction (Found::holds):
//       LoadGlobal: its row says the result is the item of the namespace
//         that has the name; the executor's hook records that item
//         (reads::global); asked: global_found;
//       an instruction whose row records its own reads before it runs and
//         says its result is an object those reads pin (the import rows):
//         asked: pinned, that some recorded item expects that very object;
//       BINARY_SUBSCR of an exact dict that is itself known by a known
//         exact str or int: the executor's hook records the item
//         (reads::subscript); asked: item_found. (It has no row: the
//         instruction stays a reason, its result is known.)
//     NOT in the list, with why: LoadDeref (the cell is the body frame's
//     own, made when the frame began and before the block, or one of the
//     function object's closure; nothing ties it to a later arrival yet);
//     an attribute of a module or an object, a method, a truth test, a
//     membership test (no row says what the lookup resolves to);
//   - NEUTRAL names every kind of instruction that a recorded run may run
//     with nothing said in the list: the kinds that only move values inside
//     the run's own frame or make a new exact container of their operands.
//     A kind that is not named is a reason;
//   - Reasons keeps every distinct reason in the order they were met, with
//     counts; the first is the one a status names.
// What it may not assume
//   - that a value of an immutable class is the same on a later run.

use std::collections::{HashMap, HashSet};

use pyo3::ffi::{self, PyObject};

use crate::cfg::ir::{Op, ValueId};
use crate::cfg::opcodes::*;
use crate::exec::ffi::{tr_frame_builtins, tr_frame_globals};
use crate::exec::reads;
use crate::exec::rules::{Enters, Read, Res, Row, Site};

// A list that rests on a read which reads.rs holds: the reads are its
// checks, and until a later arrival is entered under them the list is not
// shown to be the body.
pub const PENDING: &str = "the reads of the run are recorded, and nothing runs them as checks on a later arrival yet";

// What is asked behind an instruction of one of the kinds of FOUND (see the
// head of the file). The objects are borrowed: the frame, its code and the
// instruction's operands hold them while the instruction is in flight.
pub enum Found {
    // The frame's globals and builtins, and the name.
    Global(*mut PyObject, *mut PyObject, *mut PyObject),
    Pinned,
    // The dict and the key.
    Item(*mut PyObject, *mut PyObject),
}

impl Found {
    // None: `op` is of no listed kind, or (a subscript) its dict or its key
    // is not known.
    pub unsafe fn of(op: &Op, row: Option<&Row>, site: &Site, args: &[ValueId], depth: u32, known: &Known) -> Option<Found> {
        unsafe {
            match (op, row) {
                (Op::LoadGlobal(n), Some(row)) if row.result == Res::Pinned && matches!(row.reads, [Read::Global]) => Some(Found::Global(tr_frame_globals(site.frame), tr_frame_builtins(site.frame), *site.names.add(*n as usize))),
                (_, Some(row)) if row.result == Res::Pinned && row.before.is_some() => Some(Found::Pinned),
                (Op::Bytecode { opcode: BINARY_SUBSCR, .. }, None) => {
                    let (dict, key) = (*site.operands.first()?, *site.operands.get(1)?);
                    let plain = !dict.is_null() && !key.is_null() && ffi::PyDict_CheckExact(dict) != 0 && (ffi::PyUnicode_CheckExact(key) != 0 || ffi::PyLong_CheckExact(key) != 0);
                    (plain && args.iter().take(2).all(|x| known.lit((depth, x.0)))).then_some(Found::Item(dict, key))
                }
                _ => None,
            }
        }
    }

    // Whether reads.rs holds the read that gave `result`.
    pub unsafe fn holds(&self, result: *mut PyObject) -> bool {
        unsafe {
            match *self {
                Found::Global(globals, builtins, name) => reads::global_found(globals, builtins, name, result),
                Found::Pinned => reads::pinned(result),
                Found::Item(dict, key) => reads::item_found(dict, key, result),
            }
        }
    }
}

// An instruction that only moves a value: the constant, the load of a local
// slot, the store of a value into a slot, the slot emptied.
pub enum Plain {
    Const,
    Load(u32),
    Store(u32, u32),
    Delete(u32),
}

impl Plain {
    pub fn of(op: &Op, args: &[ValueId]) -> Option<Plain> {
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
    // The body's activation, how many params its block has, the local
    // slots of its frame that the run changed, and value -> the position
    // (among the params, then the frame's locals) of the input it is.
    body: u32,
    nparams: u32,
    changed: HashSet<u32>,
    inputs: HashMap<u32, u32>,
}

impl Known {
    // A run begins whose body is activation `body`, on a block with these
    // param values.
    pub fn begin(body: u32, params: &[u32]) -> Known {
        Known { body, nparams: params.len() as u32, inputs: params.iter().enumerate().map(|(i, v)| (*v, i as u32)).collect(), ..Known::default() }
    }

    pub fn plain(&mut self, depth: u32, v: u32, op: Plain) {
        match op {
            Plain::Const => drop(self.lit.insert((depth, v))),
            Plain::Load(slot) => {
                if self.slots.contains(&(depth, slot)) {
                    self.lit.insert((depth, v));
                } else {
                    self.makers.insert((depth, v), "a local the run did not store a known value in (an input, a parameter)".into());
                    if depth == self.body && !self.changed.contains(&slot) {
                        self.inputs.insert(v, self.nparams + slot);
                    }
                }
            }
            Plain::Store(slot, x) => {
                self.changes(depth, slot);
                if self.lit.contains(&(depth, x)) {
                    self.slots.insert((depth, slot));
                } else {
                    self.slots.remove(&(depth, slot));
                }
            }
            Plain::Delete(slot) => {
                self.changes(depth, slot);
                self.slots.remove(&(depth, slot));
            }
        }
    }

    // An instruction of activation `depth` puts something else into the
    // local `slot` of its frame.
    pub fn changes(&mut self, depth: u32, slot: u32) {
        if depth == self.body {
            self.changed.insert(slot);
        }
    }

    // The position of the input that the value `at` is: a param of the
    // block, or a local of the body's frame loaded before the run changed
    // that slot.
    pub fn input(&self, at: (u32, u32)) -> Option<u32> {
        if at.0 == self.body { self.inputs.get(&at.1).copied() } else { None }
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
