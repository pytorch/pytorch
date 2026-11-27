// The recorder of call lists, in SHADOW mode (TORCH_RUST_CALLS=shadow; unset
// or anything else: on() is false and nothing here runs). It looks on while
// one ordinary run of a block goes by, the run before the one today's engine
// traces, and says what list of the body's own C calls it would have made.
// It changes nothing a program computes: no instruction runs differently,
// nothing of the program is called, no tensor is held (origins.rs), nothing
// it finds is used by the engine. What the recorded run reads is recorded
// as a trace's reads are (reads.rs, which holds what was read from until the
// recording ends or stops, as it does for a trace).
//
// What it is given (by run.rs and mod.rs's eval_frame)
//   - begin / end around the run of the block's plain instructions; scope
//     around each activation (the body, a frame under it); frame for each
//     Python frame entered, and whether the executor runs it;
//   - before each instruction: what it is and its operands, as covered.rs is
//     given them; after it: whether it raised and its results; tick for an
//     instruction of the executor's inline path.
// What it guarantees
//   - a recording begins only under a neutral ambient state and with the
//     reads of the thread its own (reading.rs): the activations of the run
//     record what they read (Act::reading), and reads() says so to run.rs.
//     A trace of today's engine that begins while the recording has them
//     takes them, and the recording stops ("a trace began under the
//     recording"); so does one whose reads met something no check covers;
//   - THE DEFAULT IS A STOP: the recording ends, with the instruction and
//     the reason, at whatever is not understood: no row (the rule table's
//     reason), an operand with no origin (a tensor no recorded read leads
//     to is one), a result that is no tensor, an
//     operator the dispatcher saw under an instruction that is no entry, a
//     frame the executor does not run, a frame under an entry's call, a
//     call that raised, an instruction the optimizer made;
//   - an ENTRY exists only for a C callee with a row whose result is an
//     entry (rules::license), and for an operator, a subscript or an
//     attribute instruction on a plain tensor that a line of the license of
//     slots.rs takes; its operands are templates from their origins;
//   - what the rule table has not taken over (NotMine) is neither: it is
//     counted ("left to today's code"), apart where it was handed a tensor;
//   - releases are what was seen: a result that dies behind the entry that
//     names it is that entry's move, any other death a drop where it
//     happened; a result whose death was not seen is dropped at the end of
//     the body and counted (late);
//   - a finished recording goes through List::new, the one check of lists,
//     and, if every entry is pure and the witness of none disagreed with
//     what the entry says, is run and compared (shadow.rs), here and on later
//     arrivals at the place (again.rs);
//   - A RECORDING THAT REACHED ITS END IS THE BODY ONLY IF NOTHING SAYS
//     OTHERWISE, AND EVERYTHING NOT UNDERSTOOD SAYS OTHERWISE: each of these
//     is a reason (Shadow::reasons), and a Shadow with one is "partial":
//     an instruction that is no entry, no kind of known::neutral and no key
//     fact of a tensor whose key the list compares; an operand taken by
//     value that is no constant of the code; a conditional jump under the
//     body on a value that is not known; a pin (the reads that lead to it
//     are recorded, and nothing runs them as a check on a later arrival
//     yet); an output that the list does not state;
//     a release that was not seen; an operator between the instructions;
//     an exception raised under the body; a generator's yield; a witness
//     that disagrees with its entry;
//   - a Python frame is the body's only if an instruction that may enter
//     one (no row, or a row that names frames) is in flight in the innermost
//     activation of the run. Any other frame (a finalizer, a weak-reference
//     callback, a signal handler: no instruction called it) is a STOP;
//   - nothing is witnessed behind a stop: the entry's witness, if its call
//     is in flight, and the run's are ended there;
//   - every recording is kept: one that begins inside a stopped one (a
//     block of a frame entered later) is recorded on top of it, and both end
//     where their blocks end.
// What it may not assume
//   - that a value of an immutable class is a constant: it is taken by value
//     and, unless it is one of the code, the list is partial;
//   - that the reads stand because they began: standing() is asked before
//     every instruction and at the end.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::LazyLock;

use pyo3::ffi::{self, PyObject};
use pyo3::prelude::*;

use super::again::{self, Armed};
use super::entries::{blank, text, Call, Maker};
use super::known::{self, Reasons};
pub use super::known::Plain;
use super::origins::{class_text, plain_tensor, Origin, Origins};
use super::shadow::{self, Shadow};
use super::{reading, slots, witness, Assert, Callee, Entry, Tmpl, PURE};
use crate::cfg::ir::{Instr, Op};
use crate::cfg::opcodes::IS_OP;
use crate::exec::ffi::{tr_frame_localsplus, tr_frame_nlocalsplus, tr_frame_qualname, Frame};
use crate::exec::rules::{self, Class, Enters, License, Res, Site};
use crate::exec::tensor_key::TensorKey;
use crate::exec::hist;
use crate::exec::reads;

static ON: LazyLock<bool> = LazyLock::new(|| std::env::var_os("TORCH_RUST_CALLS").is_some_and(|v| v == "shadow"));

#[inline]
pub fn on() -> bool {
    *ON
}

thread_local! {
    static LIVE: Cell<bool> = const { Cell::new(false) };
    static DEPTH: Cell<u32> = const { Cell::new(0) };
    // The recordings in flight, innermost last: all but the last stopped.
    static REC: RefCell<Vec<Rec>> = const { RefCell::new(Vec::new()) };
    // The places (plan, block) a run was recorded of.
    static TAKEN: RefCell<std::collections::HashSet<(usize, u32)>> = RefCell::new(std::collections::HashSet::new());
}

// Whether the run being recorded on this thread records what it reads: an
// activation of it is to tell reads.rs (Act::reading).
pub fn reads() -> bool {
    LIVE.get() && REC.with_borrow(|r| r.last().is_some_and(|r| r.reads))
}

// Whether a run of the place with this key was recorded on this thread.
pub fn taken(key: (usize, u32)) -> bool {
    TAKEN.with_borrow(|t| t.contains(&key))
}

// Whether a run is being recorded on this thread and has not stopped.
#[inline]
pub fn recording() -> bool {
    LIVE.get()
}

enum Pending {
    Entry(Box<Call>),
    // No entry: what it is; whether a tensor it leaves was made by it; why
    // it is no entry though it runs C code on a tensor, if that is so; the
    // tensors among its operands; whether it may enter a Python frame;
    // whether its result is a known value (None: known.rs was told).
    Other { what: String, makes: bool, why: Option<String>, operands: Vec<usize>, enters: bool, known: Option<bool> },
}

// An instruction in flight: with the operators the run's witness had seen
// and claimed before it, its activation and its value.
struct InFlight {
    pending: Pending,
    ops: usize,
    claimed: usize,
    at: (u32, u32),
}

pub(super) struct Rec {
    pub(super) shadow: Shadow,
    pub(super) origins: Origins,
    pub(super) entries: Vec<Entry>,
    stack: Vec<InFlight>,
    // The entry whose operands are being released; slots that died elsewhere.
    window: Option<usize>,
    pub(super) drops: Vec<u32>,
    // Slots of tuple results: their release is never seen.
    pub(super) tuples: Vec<u32>,
    at: u32,
    // Operators the run's witness saw under entries.
    pub(super) claimed: usize,
    pub(super) pure: bool,
    pub(super) left: HashMap<String, u64>,
    pub(super) handed: HashMap<String, u64>,
    pub(super) reasons: Reasons,
    // The witness of an entry disagreed with what the entry says.
    pub(super) disagreed: bool,
    // The witness of the entry in flight is open; what it saw, if a stop
    // ended it; what the run's saw, once it is ended.
    entry_open: bool,
    entry_seen: Option<witness::Witnessed>,
    outer: Option<witness::Witnessed>,
    // The reads of the thread are this recording's (reads.rs Owner).
    reads: bool,
}

impl Rec {
    fn maker(&mut self) -> Maker<'_> {
        Maker { origins: &mut self.origins, facts: &mut self.shadow.facts, at: self.at, depth: DEPTH.get() }
    }

    pub(super) fn stop(&mut self, at: &str, why: String) -> bool {
        if self.shadow.stop.is_none() {
            self.shadow.stop = Some((at.to_string(), why));
            self.shadow.entries = self.entries.iter().filter(|e| !matches!(e.callee, Callee::Drop(_))).count();
            self.shadow.pins = self.origins.pins.len() as u32;
            LIVE.set(false);
            if std::mem::take(&mut self.reads) {
                reading::stopped();
            }
            // Nothing is witnessed from here on.
            if std::mem::take(&mut self.entry_open) {
                self.entry_seen = Some(witness::end());
            }
            self.outer = Some(witness::end());
        }
        false
    }

    fn push(&mut self, pending: Pending, at: (u32, u32)) -> bool {
        self.stack.push(InFlight { pending, ops: witness::top().0, claimed: self.claimed, at });
        true
    }

    // Whether the tensor `o` is one whose key the list compares (an input)
    // or asserts (a result).
    unsafe fn keyed(&self, o: *mut PyObject) -> bool {
        unsafe { plain_tensor(o) && matches!(self.origins.get(o), Some(Origin::In(_) | Origin::Res(_) | Origin::Part(..))) }
    }

    fn baked(&mut self) {
        for what in std::mem::take(&mut self.origins.baked) {
            self.reasons.add(format!("an operand taken by value that is no constant of the code: {what}"));
        }
    }

    // The deaths since the last instruction: of a slot the entry just made
    // names once, whole, that entry's move; else a drop here.
    pub(super) fn settle(&mut self) {
        for j in self.origins.deaths() {
            let moved = self.window.is_some_and(|k| {
                let mut leaves = Vec::new();
                self.entries[k].args.iter().for_each(|t| t.leaves(&mut leaves));
                let named = leaves.into_iter().filter(|t| matches!(t, Tmpl::Res(s) | Tmpl::Move(s) | Tmpl::Part(s, _) if *s == j)).count();
                named == 1 && self.entries[k].args.iter_mut().any(|t| super::entries::moves(t, j))
            });
            if !moved {
                self.drops.push(j);
            }
        }
        self.window = None;
    }

    pub(super) fn flush(&mut self) {
        if !self.drops.is_empty() {
            self.entries.push(blank(Callee::Drop(std::mem::take(&mut self.drops).into()), self.at));
        }
    }

    unsafe fn before(&mut self, body: bool, v: u32, ins: &Instr, site: &Site, callee: Option<*mut PyObject>) -> bool {
        unsafe {
            self.settle();
            self.shadow.seen += 1;
            if self.shadow.stop.is_some() {
                self.shadow.left_behind += 1;
                return false;
            }
            if body {
                self.at = ins.origin;
            }
            let (all, depth) = (site.operands, DEPTH.get());
            let at = (depth, v);
            let callee = callee.filter(|f| !f.is_null());
            let what = rules::ops::what(&ins.op, callee);
            let name = callee.map_or_else(|| hist::kind(&ins.op).to_string(), |f| format!("call {}", hist::callee_name(f)));
            if let Err(why) = reads::standing() {
                return self.stop(&name, why);
            }
            if matches!(ins.op, Op::Aten { .. } | Op::Lit(_) | Op::Pinned(_) | Op::Expect(_) | Op::Region(_) | Op::Autograd(_)) {
                return self.stop(&name, "an instruction the optimizer made: the run is not on plain instructions".into());
            }
            let classes: Vec<Class> = all.iter().map(|o| Class::of(*o)).collect();
            let row = match rules::license(&what, &classes, site) {
                License::NoRow(why) => return self.stop(&name, why),
                License::Row(row) => Some(row),
                License::NotMine => None,
            };
            if let Some(made) = self.maker().built(ins, all) {
                self.origins.built(at, made);
                return self.push(Pending::Other { what: name, makes: false, why: None, operands: Vec::new(), enters: false, known: Some(false) }, at);
            }
            let facts = self.shadow.facts;
            let call = match (row, callee) {
                (Some(row), Some(f)) if row.result == Res::Entry => Some(self.maker().vector(f, ins, all, site.kwnames)),
                (None, None) => self.maker().slot(ins, site),
                _ => None,
            };
            let why = match call {
                Some(Ok(mut call)) => {
                    self.flush();
                    self.baked();
                    call.operands.iter().enumerate().for_each(|(i, o)| shadow::tensors(*o, i, &mut call.ids));
                    self.push(Pending::Entry(Box::new(call)), at);
                    witness::begin();
                    self.entry_open = true;
                    return true;
                }
                // A C callee with a row is an entry or nothing.
                Some(Err(why)) if callee.is_some() => return self.stop(&name, why),
                Some(Err(why)) => Some(why),
                None => None,
            };
            self.origins.baked.clear();
            self.origins.used(depth, ins.args.iter().map(|x| x.0));
            let first = if all.first().is_some_and(|o| o.is_null()) { 2 } else { 1 };
            let given = callee.map_or(all, |_| &all[first.min(all.len())..]);
            let tensor = given.iter().any(|o| plain_tensor(*o));
            let later = rules::later(&what).map_or(String::new(), |family| format!(" (later: {family})"));
            let class = given.first().map_or("-".into(), |o| class_text(*o));
            // A key fact of a tensor (entries.rs counted it): what a line of
            // the license of slots.rs reads off the key.
            let fact = self.shadow.facts > facts;
            let plain = Plain::of(&ins.op, &ins.args);
            let known = match (&ins.op, &plain) {
                (_, Some(_)) => None,
                (Op::IsNone, _) => Some(self.origins.known.lit((depth, ins.args[0].0)) || self.keyed(all[0])),
                (Op::Bytecode { opcode: IS_OP, .. }, _) => Some(ins.args.iter().all(|x| self.origins.known.lit((depth, x.0)))),
                _ => Some(fact && self.keyed(all[0])),
            };
            if let Some(plain) = plain {
                self.origins.known.plain(depth, v, plain);
            }
            match row {
                None if fact => {}
                None => {
                    *self.left.entry(format!("{name} x {class}")).or_default() += 1;
                    let python = callee.is_some_and(|f| [&raw mut ffi::PyFunction_Type, &raw mut crate::exec::ffi::PyMethod_Type].contains(&ffi::Py_TYPE(f)));
                    if tensor && !python {
                        *self.handed.entry(format!("{name}{later}")).or_default() += 1;
                    }
                    self.reasons.add(format!("no row: {name}{later} x {class}"));
                }
                Some(row) if known::neutral(&ins.op, row).is_some() => {}
                Some(row) => self.reasons.add(format!("a read with no check, or a kind the recorder does not list: {name} (its row reads {:?})", row.reads)),
            }
            // C code of torch that is no entry makes what it returns; so does
            // an instruction that runs C code on a tensor.
            let makes = callee.map_or(row.is_none() && tensor, |f| rules::torch::name(f).is_some());
            let operands = all.iter().filter(|o| plain_tensor(**o)).map(|o| *o as usize).collect();
            let enters = row.is_none_or(|r| r.enters == Enters::Named);
            let what = if matches!(ins.op, Op::ToBool) { format!("{name} of {}", self.origins.known.maker((depth, ins.args[0].0))) } else { format!("{name}{later}") };
            self.push(Pending::Other { what, makes, why, operands, enters, known }, at)
        }
    }

    unsafe fn after(&mut self, ok: bool, results: &[*mut PyObject]) {
        unsafe {
            let Some(InFlight { pending, ops, claimed, at }) = self.stack.pop() else { return };
            let seen = match pending {
                Pending::Entry(_) if std::mem::take(&mut self.entry_open) => Some(witness::end()),
                Pending::Entry(_) => self.entry_seen.take(),
                Pending::Other { .. } => None,
            };
            if self.shadow.stop.is_some() {
                return;
            }
            let (now, last) = witness::top();
            // Operators the dispatcher saw under this instruction and under no
            // entry inside it.
            let direct = now.saturating_sub(ops).saturating_sub(self.claimed - claimed);
            match pending {
                Pending::Other { what, makes, why, operands, known, .. } => {
                    if let Some(known) = known {
                        self.origins.known.set(at, known && ok, || format!("the result of {what}"));
                    }
                    if direct > 0 {
                        let ran = format!("{direct} operator(s) ran under an instruction that is no entry (the last: {})", last.unwrap_or_default());
                        self.stop(&what, why.map_or(ran.clone(), |why| format!("{why}; {ran}")));
                    } else if makes {
                        // What it hands back of its operands it did not make.
                        for r in results.iter().filter(|r| plain_tensor(**r) && !operands.contains(&(**r as usize))) {
                            if self.origins.get(*r).is_none() {
                                self.origins.mark(*r, Origin::Unknown(what.clone()));
                            }
                        }
                    }
                }
                Pending::Entry(call) => {
                    self.claimed += now.saturating_sub(ops);
                    let Call { mut entry, label, ids, operands } = *call;
                    let k = self.entries.len() as u32;
                    let notes = shadow::disagreements(&seen.unwrap_or_default(), &ids, &entry.flags.writes);
                    // A list with such an entry is never run again: the row
                    // it rests on did not say what the call did.
                    self.disagreed |= !notes.is_empty();
                    notes.iter().for_each(|n| self.reasons.add(format!("the witness of an entry disagrees with it: {label}: {n}")));
                    self.shadow.notes.extend(notes.into_iter().map(|n| format!("{label}: {n}")));
                    if !ok {
                        self.stop(&label, "the call of an entry raised: which handler takes it is a later unit".into());
                        return;
                    }
                    let result = results.first().copied().unwrap_or(std::ptr::null_mut());
                    let class = |o: *mut PyObject| Some(Py::from_borrowed_ptr(Python::assume_attached(), ffi::Py_TYPE(o) as *mut PyObject));
                    if result.is_null() || result == ffi::Py_None() {
                        entry.assert = (!result.is_null()).then(|| Assert { class: class(result), ..Assert::default() });
                    } else if plain_tensor(result) && self.origins.get(result).is_some() {
                        let Some(same) = operands.iter().position(|o| *o == result) else {
                            self.stop(&label, "the result is a tensor of the list that is no operand of the call".into());
                            return;
                        };
                        entry.assert = Some(Assert { same: Some(same as u32), ..Assert::default() });
                    } else if plain_tensor(result) {
                        let key = TensorKey::of(result);
                        if key.is_none() || !self.origins.mark(result, Origin::Res(k)) {
                            self.stop(&label, "a result that has no key, or that no weak reference can be made to".into());
                            return;
                        }
                        (entry.out, entry.assert) = (Some(k), Some(Assert { key, class: class(result), ..Assert::default() }));
                    } else if slots::tuple_like(result) {
                        let n = ffi::PyTuple_GET_SIZE(result);
                        for i in 0..n {
                            let item = ffi::PyTuple_GET_ITEM(result, i);
                            if !plain_tensor(item) || self.origins.get(item).is_some() || !self.origins.mark(item, Origin::Part(k, i as u32)) {
                                self.stop(&label, format!("a result tuple that holds a {}, or a tensor of the list", class_text(item)));
                                return;
                            }
                        }
                        (entry.out, entry.assert) = (Some(k), Some(Assert { class: class(result), count: Some(n as usize), ..Assert::default() }));
                        self.tuples.push(k);
                    } else {
                        self.stop(&label, format!("the result of an entry is a {}: results that are no tensors are a later unit", class_text(result)));
                        return;
                    }
                    self.pure &= entry.flags.bits & PURE != 0;
                    self.window = Some(self.entries.len());
                    self.entries.push(entry);
                }
            }
        }
    }
}

fn with<R: Default>(f: impl FnOnce(&mut Rec) -> R) -> R {
    let done = REC.with_borrow_mut(|r| r.last_mut().map_or_else(R::default, f));
    // (What a stop let go of, outside the borrow.)
    reading::release();
    done
}

// A run of the block `place` (key: its plan and block) begins on `params`
// in the frame `frame`. A recording in flight has stopped (shadow_due):
// this one is recorded on top of it.
pub unsafe fn begin(place: String, key: (usize, u32), params: &[*mut PyObject], frame: *mut Frame) {
    unsafe {
        TAKEN.with_borrow_mut(|t| t.insert(key));
        let locals = tr_frame_localsplus(frame);
        let origins = Origins::begin(params.iter().copied().chain((0..tr_frame_nlocalsplus(frame) as usize).map(|i| *locals.add(i))));
        let shadow = Shadow { place, key, ..Shadow::default() };
        witness::begin();
        let mut rec = Rec { shadow, origins, entries: Vec::new(), stack: Vec::new(), window: None, drops: Vec::new(), tuples: Vec::new(), at: 0, claimed: 0, pure: true, left: HashMap::new(), handed: HashMap::new(), reasons: Reasons::default(), disagreed: false, entry_open: false, entry_seen: None, outer: None, reads: false };
        // No recording of a run that a list could not be entered under, and
        // none whose reads are not recorded.
        let began = reading::begin();
        LIVE.set(true);
        match began {
            Ok(()) => rec.reads = true,
            Err(why) => drop(rec.stop("the start of the body", why)),
        }
        REC.with_borrow_mut(|r| r.push(rec));
    }
}

pub unsafe fn end(outs: Option<&[*mut PyObject]>) {
    unsafe {
        LIVE.set(false);
        let Some(mut rec) = REC.with_borrow_mut(|r| r.pop()) else { return };
        let outer = rec.outer.take().unwrap_or_else(witness::end);
        // The block's exception, if it left by one, stays as it is.
        let raised = ffi::PyErr_GetRaisedException();
        let key = rec.shadow.key;
        // Whether every read of the run was recorded, asked before they are
        // let go of; then what they held is released, as early as it can be.
        let stood = if std::mem::take(&mut rec.reads) { reading::end() } else { None };
        reading::release();
        if let Some(why) = stood {
            rec.stop("the end of the body", why);
        }
        let (shadow, armed) = rec.end(outs, outer);
        // What ran under this recording ran behind the stop of the one around it.
        let seen = shadow.seen;
        with(|r| (r.shadow.seen, r.shadow.left_behind) = (r.shadow.seen + seen, r.shadow.left_behind + seen));
        let kept = shadow::keep(shadow);
        if let Some(armed) = armed {
            again::arm(key, Armed { kept, ..armed });
        }
        if !raised.is_null() {
            ffi::PyErr_SetRaisedException(raised);
        }
    }
}

// (told, entry). told: after() is to be told how the instruction ended.
// entry: the instruction is an entry, so every operand of it is a template
// of the list and nothing it is handed is a read of the run: the caller
// runs it with the reads off (what C code is handed is held by reads.rs
// until the recording ends, and a list the body built for the call would
// keep its tensors alive past the call).
pub unsafe fn before(body: bool, v: u32, ins: &Instr, site: &Site, callee: Option<*mut PyObject>) -> (bool, bool) {
    with(|r| unsafe {
        let told = r.before(body, v, ins, site, callee);
        (told, told && r.entry_open)
    })
}

pub unsafe fn after(ok: bool, results: &[*mut PyObject]) {
    with(|r| unsafe { r.after(ok, results) })
}

// An instruction of the executor's inline path, which only moves the value
// `v`: what it is.
pub fn tick(v: u32, plain: Plain) {
    with(|r| {
        r.settle();
        r.shadow.seen += 1;
        r.shadow.left_behind += r.shadow.stop.is_some() as u64;
        if let Plain::Store(_, x) = plain {
            r.origins.used(DEPTH.get(), std::iter::once(x));
        }
        r.origins.known.plain(DEPTH.get(), v, plain);
    })
}

// A conditional jump of the recorded run tested the value `v`.
pub fn jump(v: u32) {
    with(|r| {
        if r.shadow.stop.is_none() {
            let at = (DEPTH.get(), v);
            if r.origins.known.lit(at) {
                r.shadow.jumps.0 += 1;
            } else {
                r.shadow.jumps.1 += 1;
                let why = format!("a jump under the body tested {}", r.origins.known.maker(at));
                r.reasons.add(why);
            }
        }
    })
}

// Something of the recorded run that no instruction told of: a generator's
// yield, an exception on its way to a handler.
pub fn unseen(what: &str) {
    with(|r| {
        if r.shadow.stop.is_none() {
            r.reasons.add(what.to_string());
        }
    })
}

// An activation of the recorded run begins, or is over.
pub fn scope(begins: bool) {
    if begins {
        DEPTH.set(DEPTH.get() + 1);
    } else {
        with(|r| {
            r.settle();
            r.origins.left(DEPTH.get());
        });
        DEPTH.set(DEPTH.get().saturating_sub(1));
    }
}

// A Python frame is entered under the recorded run; `runs`: the executor
// runs it (asked first), false: it is left to CPython.
pub unsafe fn frame(frame: *mut Frame, runs: bool) {
    if runs {
        // What reads.rs is told of an instruction: whether it entered a frame.
        reads::frame_entered();
    }
    // (No release here, where the frame is not running yet: the next thing
    // the recorder is told, or the next recording's begin, lets go.)
    REC.with_borrow_mut(|r| r.last_mut().map(|r| unsafe {
        if r.shadow.stop.is_some() {
            return;
        }
        let name = || text(tr_frame_qualname(frame));
        let depth = DEPTH.get();
        if !runs {
            r.stop(&name(), "a frame the executor does not run: nothing of it is seen".into());
            return;
        }
        let stop = match r.stack.last() {
            Some(InFlight { pending: Pending::Entry(call), .. }) => Some((call.label.clone(), format!("the Python frame {} ran under the C call of an entry", name()))),
            // The innermost activation has an instruction in flight that may
            // enter a frame: the frame is taken for the one it called.
            Some(InFlight { pending: Pending::Other { enters: true, .. }, at, .. }) if at.0 == depth => None,
            Some(InFlight { pending: Pending::Other { what, .. }, at, .. }) if at.0 == depth => Some((name(), format!("a frame that no instruction of the body called: it ran under {what}, which enters none"))),
            _ => Some((name(), "a frame that no instruction of the body called: it ran between two instructions (a finalizer, a callback)".into())),
        };
        match stop {
            Some((at, why)) => drop(r.stop(&at, why)),
            None => r.shadow.frames += 1,
        }
    }));
}

// Whether the list of the place `key` is to run on this arrival (again.rs).
pub fn again_due(key: (usize, u32)) -> bool {
    !LIVE.get() && again::armed(key)
}

pub use again::{begin as again_begin, outs as again_outs, Again};

// Today's engine traces the place with this key.
pub fn traced(key: (usize, u32)) {
    shadow::traced(key);
}

// A trace of today's engine is about to begin on this thread: a recording
// that has the reads stops, and what it recorded of them is let go of before
// the trace begins its own.
pub fn yields() {
    with(|r| {
        if r.reads {
            r.stop("a trace of the engine", "a trace began under the recording".into());
        }
    })
}
