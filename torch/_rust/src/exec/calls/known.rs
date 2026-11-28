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
//   - A VALUE IS KNOWN IN ONE OF TWO KINDS (Kind), and whoever asks says
//     which it needs:
//       Value   its exact class and its value (the bits of a float, the
//               digits of an int, the content of a str): what a later
//               arrival is held to by exec/exact.rs. Which OBJECT it is, is
//               not known: two floats of the same bits are two objects;
//       Object  which object it is, on every arrival the list is entered
//               on: a constant of the code object (the code holds it), an
//               object a recorded read expects by identity, one of the
//               objects there is exactly one of (None, True, False,
//               Ellipsis, NotImplemented: singleton()). It implies Value.
//     Produced: Object by a constant, by IsNone and by `is` (a bool), by a
//     read reads.rs holds by identity; Value by an input the list is
//     specialized on, by a read held by value (an exact int, float or str),
//     by a key fact of a tensor; Object for any of these if the object is a
//     singleton. A load or a store of a local keeps the kind, and so does
//     the instruction that names one result of an instruction that hands
//     back several (PARTS).
//     Consumed: an operand taken by value (origins.rs), the test of a
//     jump (a bool), IsNone (None is the one object of its class) and the
//     key of a dict item ask for Value; `is` / `is not` and the dict of a
//     dict item ask for Object: which object an operand is cannot be told
//     from its value; the callee of an entry asks for Object too: the list
//     calls the recorded callee, and only a callee that is the same object
//     on every arrival is the one the body calls (record.rs). Nothing else
//     of the neutral kinds asks about a value;
//   - lit() is true only for a value that is a constant of the code object,
//     a copy of one through locals the run itself stored, one record.rs
//     set as a fact that the list checks (the key of an input or a result),
//     an input that the list was specialized on (input(): a param of the
//     block, or a local of the body's frame that the run had not stored
//     into when it loaded it; origins.rs keeps the value, and a later
//     arrival is asked for exactly it), or the result of an instruction of
//     one of the kinds of FOUND for which reads.rs holds the read that gave
//     it: that read is one of the list's checks, settled when the run is
//     over and asked before the list runs on a later arrival (reading.rs,
//     again.rs). A parameter of a frame under the body, a default, the content of
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
//       a truth test of an exact dict, list, set or OrderedDict that is
//         itself known as an object: its row says the result is a function
//         of the container's length and records that length before the
//         test runs (reads::truth); asked: length_found;
//     NOT in the list, with why: LoadDeref (no row says its result is what
//     the read of the cell found; a cell the body's frame made is another
//     on every call, and reads of it make the reads no check: reading.rs);
//     an attribute of a module or an object, a method, a membership test
//     (no row says what the lookup resolves to);
//   - PARTS. A method load hands back two values: (the function, the
//     receiver) or (nothing, the attribute). What is known of each is kept
//     by its place (parted) and taken by the instruction that names it
//     (project): the function or the attribute as the load is known (one
//     of the kinds of FOUND, asked about the first of the two that is
//     there), the receiver as it was known as an operand. Of any other
//     instruction that hands back several, nothing is known of any;
//   - TRUTH, what the row of a truth test (ToBool, UNARY_NOT) says its
//     result is known by, taken from the row's columns and from nothing
//     else (Truth::of; a row of any other shape is a reason, as for every
//     kind that is not listed):
//       reads nothing, result Lit (an immutable): a function of the
//         operand's value: known when the operand is known, in either kind;
//       reads Identity, result Lit (an exact tuple, frozenset or range,
//         which holds what it held when it was made): known when it is
//         known WHICH OBJECT the operand is. (A tuple known by its value
//         only, an input the list is specialized on, has the same length
//         on a later arrival too; the row's word is identity, and the
//         row's word is what is taken);
//       reads Length, result Lit: the kind of FOUND above;
//       reads Class, enters the frame it names, result Return (a Python
//         __bool__): the frame is the instruction's (frames.rs asks the
//         table which function), its own instructions are asked about one
//         at a time, and what it returns is NOT known: no origin is carried
//         out of a frame.
//     The result of a truth test is a bool: one object per value, so a
//     known one is known as an object;
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

// What is asked behind an instruction of one of the kinds of FOUND (see the
// head of the file). The objects are borrowed: the frame, its code and the
// instruction's operands hold them while the instruction is in flight.
pub enum Found {
    // The frame's globals and builtins, and the name.
    Global(*mut PyObject, *mut PyObject, *mut PyObject),
    Pinned,
    // The dict and the key.
    Item(*mut PyObject, *mut PyObject),
    // The container a truth test measured, and whether the test is the
    // negation (UNARY_NOT).
    Length(*mut PyObject, bool),
}

// What the row of a truth test says its result is known by (see TRUTH at
// the head of the file).
#[derive(Clone, Copy, PartialEq)]
pub enum Truth {
    // The operand, known at least in this kind.
    Operand(Kind),
    // The length of the operand, as a recorded read found it.
    Length,
    // What the function the row names returns.
    Returned,
}

impl Truth {
    // None: `op` is no truth test, or its row is of no listed shape.
    pub fn of(op: &Op, row: &Row) -> Option<Truth> {
        if !matches!(op, Op::ToBool | Op::Bytecode { opcode: UNARY_NOT, .. }) {
            return None;
        }
        match (row.reads, row.result, row.enters) {
            ([], Res::Lit, Enters::Never) => Some(Truth::Operand(Kind::Value)),
            ([Read::Identity], Res::Lit, Enters::Never) => Some(Truth::Operand(Kind::Object)),
            ([Read::Length], Res::Lit, Enters::Never) => Some(Truth::Length),
            ([Read::Class], Res::Return, Enters::Named) => Some(Truth::Returned),
            _ => None,
        }
    }
}

// How a value is known (see the head of the file). Object implies Value.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Kind {
    Value,
    Object,
}

// Whether there is exactly one object of what `o` is: None, True, False,
// Ellipsis, NotImplemented. The closed list: every other class may have two
// objects of one value.
pub unsafe fn singleton(o: *mut PyObject) -> bool {
    unsafe { [ffi::Py_None(), ffi::Py_True(), ffi::Py_False(), ffi::Py_Ellipsis(), ffi::Py_NotImplemented()].contains(&o) }
}

impl Kind {
    // The kind `o` is known in, where its producer knows it by identity
    // (`object`) or by value only.
    pub unsafe fn of(o: *mut PyObject, object: bool) -> Kind {
        if object || unsafe { singleton(o) } { Kind::Object } else { Kind::Value }
    }
}

impl Found {
    // None: `op` is of no listed kind, or (a subscript) its dict or its key
    // is not known.
    pub unsafe fn of(op: &Op, row: Option<&Row>, site: &Site, args: &[ValueId], depth: u32, known: &Known) -> Option<Found> {
        unsafe {
            match (op, row) {
                (Op::LoadGlobal(n), Some(row)) if row.result == Res::Pinned && matches!(row.reads, [Read::Global]) => Some(Found::Global(tr_frame_globals(site.frame), tr_frame_builtins(site.frame), *site.names.add(*n as usize))),
                (_, Some(row)) if row.result == Res::Pinned && row.before.is_some() => Some(Found::Pinned),
                // Which container it is: a length read of another says nothing.
                (_, Some(row)) if Truth::of(op, row) == Some(Truth::Length) => {
                    let container = *site.operands.first()?;
                    (!container.is_null() && known.object((depth, args.first()?.0))).then_some(Found::Length(container, !matches!(op, Op::ToBool)))
                }
                (Op::Bytecode { opcode: BINARY_SUBSCR, .. }, None) => {
                    let (dict, key) = (*site.operands.first()?, *site.operands.get(1)?);
                    let plain = !dict.is_null() && !key.is_null() && ffi::PyDict_CheckExact(dict) != 0 && (ffi::PyUnicode_CheckExact(key) != 0 || ffi::PyLong_CheckExact(key) != 0);
                    // Which dict it is, and the value of the key.
                    (plain && known.object((depth, args.first()?.0)) && known.lit((depth, args.get(1)?.0))).then_some(Found::Item(dict, key))
                }
                _ => None,
            }
        }
    }

    // Whether reads.rs holds the read that gave `result`, and in which kind
    // that makes it known: an object, where the read expects that very
    // object; a value, where it expects an equal one.
    pub unsafe fn holds(&self, result: *mut PyObject) -> Option<Kind> {
        unsafe {
            let by_identity = match *self {
                Found::Global(globals, builtins, name) => reads::global_found(globals, builtins, name, result)?,
                Found::Pinned => reads::pinned(result).then_some(true)?,
                Found::Item(dict, key) => reads::item_found(dict, key, result)?,
                // By value: the answer is what that length makes it.
                Found::Length(container, not) => reads::length_found(container).filter(|n| (result == ffi::Py_True()) == ((*n != 0) != not)).map(|_| false)?,
            };
            Some(Kind::of(result, by_identity))
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
        Op::IsNone => "whether the operand is None, the one object of its class: the answer is known when the operand's value is",
        Op::MakeCell { .. } => "it wraps a slot of the run's own frame in a new cell",
        // A frame of that work is a frame no instruction called: a stop.
        Op::Resume | Op::EvalBreaker => "the interpreter's pending work, none of which is the body's",
        Op::Bytecode { opcode: NOP | POP_TOP | PUSH_NULL | COPY | SWAP | END_FOR | CACHE | KW_NAMES | EXTENDED_ARG, .. } => "it moves values on the frame's own stack",
        Op::Bytecode { opcode: IS_OP, .. } => "whether two operands are one object: known only when both are known as objects (Kind::Object), never from their values",
        _ => return None,
    })
}

#[derive(Default)]
pub struct Known {
    // (activation, value): a value that is known, and how.
    lit: HashMap<(u32, u32), Kind>,
    // (activation, local slot): the slot holds a known value.
    slots: HashMap<(u32, u32), Kind>,
    // (activation, value) -> what made a value that is not known.
    makers: HashMap<(u32, u32), String>,
    // (activation, value) -> what is known of each value an instruction
    // that hands back several handed back, by its place (PARTS).
    parts: HashMap<(u32, u32), Vec<Option<Kind>>>,
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
        let makers = params.iter().map(|v| ((body, *v), "a parameter of the block (an input of the run)".to_string())).collect();
        Known { body, nparams: params.len() as u32, inputs: params.iter().enumerate().map(|(i, v)| (*v, i as u32)).collect(), makers, ..Known::default() }
    }

    pub fn plain(&mut self, depth: u32, v: u32, op: Plain) {
        match op {
            // The code object holds its constants: the same object always.
            Plain::Const => drop(self.lit.insert((depth, v), Kind::Object)),
            Plain::Load(slot) => {
                if let Some(kind) = self.slots.get(&(depth, slot)) {
                    self.lit.insert((depth, v), *kind);
                } else {
                    self.makers.insert((depth, v), "a local the run did not store a known value in (an input, a parameter)".into());
                    if depth == self.body && !self.changed.contains(&slot) {
                        self.inputs.insert(v, self.nparams + slot);
                    }
                }
            }
            Plain::Store(slot, x) => {
                self.changes(depth, slot);
                match self.lit.get(&(depth, x)) {
                    Some(kind) => drop(self.slots.insert((depth, slot), *kind)),
                    None => drop(self.slots.remove(&(depth, slot))),
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

    // Whether the value `at` is known, at least by its value.
    pub fn lit(&self, at: (u32, u32)) -> bool {
        self.lit.contains_key(&at)
    }

    // Whether it is known which object the value `at` is.
    pub fn object(&self, at: (u32, u32)) -> bool {
        self.lit.get(&at) == Some(&Kind::Object)
    }

    // How the value `at` is known, if it is.
    pub fn kind(&self, at: (u32, u32)) -> Option<Kind> {
        self.lit.get(&at).copied()
    }

    // The instruction `at` handed back several values: what is known of
    // each, by its place.
    pub fn parted(&mut self, at: (u32, u32), parts: Vec<Option<Kind>>) {
        self.parts.insert(at, parts);
    }

    // The value `v` of activation `depth` names result `k` of the
    // instruction `from`: it is known as that result is, or made by what
    // made `from`.
    pub fn project(&mut self, depth: u32, v: u32, from: u32, k: u32) {
        match self.parts.get(&(depth, from)).and_then(|parts| parts.get(k as usize)).copied().flatten() {
            Some(kind) => drop(self.lit.insert((depth, v), kind)),
            None => {
                let maker = self.maker((depth, from)).to_string();
                self.makers.insert((depth, v), maker);
            }
        }
    }

    // The value `at` is known in the kind `lit`, or (None) made by `maker`.
    pub fn set(&mut self, at: (u32, u32), lit: Option<Kind>, maker: impl FnOnce() -> String) {
        match lit {
            Some(kind) => drop(self.lit.insert(at, kind)),
            None => {
                self.lit.remove(&at);
                self.makers.insert(at, maker());
            }
        }
    }

    pub fn maker(&self, at: (u32, u32)) -> &str {
        self.makers.get(&at).map_or("a value of no recorded origin", |s| s.as_str())
    }

    // An activation is over.
    pub fn left(&mut self, depth: u32) {
        self.lit.retain(|(d, _), _| *d != depth);
        self.slots.retain(|(d, _), _| *d != depth);
        self.makers.retain(|(d, _), _| *d != depth);
        self.parts.retain(|(d, _), _| *d != depth);
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
