// The recorder of call lists, in SHADOW mode (TORCH_RUST_CALLS=shadow; unset
// or anything else: on() is false and nothing here runs). It looks on while
// one ordinary run of a block goes by, the run before the one today's engine
// traces, and says what list of the body's own C calls it would have made.
// It changes nothing: no instruction runs differently, nothing of the
// program is called, no object is held (origins.rs), nothing it finds is
// used by the engine.
//
// What it is given (by run.rs and mod.rs's eval_frame)
//   - begin / end around the run of the block's plain instructions; scope
//     around each activation (the body, a frame under it); frame for each
//     Python frame entered, and whether the executor runs it;
//   - before each instruction: what it is and its operands, as covered.rs is
//     given them; after it: whether it raised and its results; tick for an
//     instruction of the executor's inline path.
// What it guarantees
//   - THE DEFAULT IS A STOP: the recording ends, with the instruction and
//     the reason, at whatever is not understood: no row (the rule table's
//     reason), an operand with no origin, a result that is no tensor, an
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
//     and, if every entry is pure, is run and compared (shadow.rs).
// What it may not assume
//   - that a value of an immutable class is a constant: it is taken by value;
//   - that a tensor with no origin was found by a read a region checks.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::LazyLock;

use pyo3::ffi::{self, PyObject};
use pyo3::prelude::*;

use super::entries::{blank, text, Call, Maker};
use super::origins::{class_text, plain_tensor, Origin, Origins};
use super::shadow::{self, Shadow};
use super::{slots, witness, Assert, Callee, Entry, List, Tmpl, PURE};
use crate::cfg::ir::{Instr, Op};
use crate::exec::ffi::{tr_frame_localsplus, tr_frame_nlocalsplus, tr_frame_qualname, Frame};
use crate::exec::rules::{self, Class, License, Res, Site};
use crate::exec::tensor_key::TensorKey;
use crate::exec::hist;

static ON: LazyLock<bool> = LazyLock::new(|| std::env::var_os("TORCH_RUST_CALLS").is_some_and(|v| v == "shadow"));

#[inline]
pub fn on() -> bool {
    *ON
}

thread_local! {
    static LIVE: Cell<bool> = const { Cell::new(false) };
    static DEPTH: Cell<u32> = const { Cell::new(0) };
    static REC: RefCell<Option<Rec>> = const { RefCell::new(None) };
    // The places (plan, block) a run was recorded of.
    static TAKEN: RefCell<std::collections::HashSet<(usize, u32)>> = RefCell::new(std::collections::HashSet::new());
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
    // it is no entry though it runs C code on a tensor, if that is so.
    Other { what: String, makes: bool, why: Option<String> },
}

struct Rec {
    shadow: Shadow,
    origins: Origins,
    entries: Vec<Entry>,
    stack: Vec<(Pending, usize, usize)>,
    // The entry whose operands are being released; slots that died elsewhere.
    window: Option<usize>,
    drops: Vec<u32>,
    // Slots of tuple results: their release is never seen.
    tuples: Vec<u32>,
    at: u32,
    // Operators the run's witness saw under entries.
    claimed: usize,
    pure: bool,
    left: HashMap<String, u64>,
    handed: HashMap<String, u64>,
}

impl Rec {
    fn maker(&mut self) -> Maker<'_> {
        Maker { origins: &mut self.origins, loose: &mut self.shadow.loose, facts: &mut self.shadow.facts, at: self.at, depth: DEPTH.get() }
    }

    fn stop(&mut self, at: &str, why: String) -> bool {
        if self.shadow.stop.is_none() {
            self.shadow.stop = Some((at.to_string(), why));
            self.shadow.entries = self.entries.iter().filter(|e| !matches!(e.callee, Callee::Drop(_))).count();
            LIVE.set(false);
        }
        false
    }

    // The deaths since the last instruction: of a slot the entry just made
    // names once, whole, that entry's move; else a drop here.
    fn settle(&mut self) {
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

    fn flush(&mut self) {
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
            let callee = callee.filter(|f| !f.is_null());
            let what = rules::ops::what(&ins.op, callee);
            let name = callee.map_or_else(|| hist::kind(&ins.op).to_string(), |f| format!("call {}", hist::callee_name(f)));
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
                self.origins.built((depth, v), made);
                self.stack.push((Pending::Other { what: name, makes: false, why: None }, witness::top().0, self.claimed));
                return true;
            }
            let call = match (row, callee) {
                (Some(row), Some(f)) if row.result == Res::Entry => Some(self.maker().vector(f, ins, all, site.kwnames)),
                (None, None) => self.maker().slot(ins, site),
                _ => None,
            };
            let why = match call {
                Some(Ok(mut call)) => {
                    self.flush();
                    call.operands.iter().enumerate().for_each(|(i, o)| shadow::tensors(*o, i, 0, &mut call.ids));
                    self.stack.push((Pending::Entry(Box::new(call)), witness::top().0, self.claimed));
                    witness::begin();
                    return true;
                }
                // A C callee with a row is an entry or nothing.
                Some(Err(why)) if callee.is_some() => return self.stop(&name, why),
                Some(Err(why)) => Some(why),
                None => None,
            };
            self.origins.used(depth, ins.args.iter().map(|x| x.0));
            let first = if all.first().is_some_and(|o| o.is_null()) { 2 } else { 1 };
            let given = callee.map_or(all, |_| &all[first.min(all.len())..]);
            let tensor = given.iter().any(|o| plain_tensor(*o));
            let later = rules::later(&what).map_or(String::new(), |family| format!(" (later: {family})"));
            if row.is_none() {
                *self.left.entry(format!("{name} x {}", given.first().map_or("-".into(), |o| class_text(*o)))).or_default() += 1;
                let python = callee.is_some_and(|f| [&raw mut ffi::PyFunction_Type, &raw mut crate::exec::ffi::PyMethod_Type].contains(&ffi::Py_TYPE(f)));
                if tensor && !python {
                    *self.handed.entry(format!("{name}{later}")).or_default() += 1;
                }
            }
            // C code of torch that is no entry makes what it returns; so does
            // an instruction that runs C code on a tensor.
            let makes = callee.map_or(row.is_none() && tensor, |f| rules::torch::name(f).is_some());
            self.stack.push((Pending::Other { what: format!("{name}{later}"), makes, why }, witness::top().0, self.claimed));
            true
        }
    }

    unsafe fn after(&mut self, ok: bool, results: &[*mut PyObject]) {
        unsafe {
            let Some((pending, ops, claimed)) = self.stack.pop() else { return };
            let seen = matches!(pending, Pending::Entry(_)).then(witness::end);
            let (now, last) = witness::top();
            if self.shadow.stop.is_some() {
                return;
            }
            // Operators the dispatcher saw under this instruction and under no
            // entry inside it.
            let direct = now.saturating_sub(ops).saturating_sub(self.claimed - claimed);
            match pending {
                Pending::Other { what, makes, why } => {
                    if direct > 0 {
                        let ran = format!("{direct} operator(s) ran under an instruction that is no entry (the last: {})", last.unwrap_or_default());
                        self.stop(&what, why.map_or(ran.clone(), |why| format!("{why}; {ran}")));
                    } else if makes {
                        for r in results.iter().filter(|r| plain_tensor(**r)) {
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

    // The recording is over: `outs`, what the block hands on (None: it left
    // by an exception).
    unsafe fn end(mut self, outs: Option<&[*mut PyObject]>, outer: witness::Witnessed) -> Shadow {
        unsafe {
            self.settle();
            let counts = |m: HashMap<String, u64>| {
                let mut v: Vec<(String, u64)> = m.into_iter().collect();
                v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                v
            };
            self.shadow.left = counts(std::mem::take(&mut self.left));
            self.shadow.handed = counts(std::mem::take(&mut self.handed));
            self.shadow.by_value = counts(self.origins.by_value.iter().map(|(k, n)| (k.to_string(), *n)).collect());
            let Some(outs) = outs.filter(|_| self.shadow.stop.is_none()) else {
                self.stop("the end of the body", "the block left by an exception".into());
                return self.shadow;
            };
            self.shadow.entries = self.entries.iter().filter(|e| !matches!(e.callee, Callee::Drop(_))).count();
            let mut outputs = Vec::new();
            let mut made = Vec::new();
            for o in outs.iter().filter(|o| plain_tensor(**o)) {
                match self.origins.get(*o) {
                    Some(Origin::Res(j)) => outputs.push(Tmpl::Res(*j)),
                    Some(Origin::Part(j, i)) => outputs.push(Tmpl::Part(*j, Box::new([*i]))),
                    _ => continue,
                }
                made.push(*o);
            }
            self.shadow.outputs = (outputs.len(), outs.len() - outputs.len());
            // An output that holds results of entries and is none itself: the
            // list does not say how it is made.
            self.shadow.unstated = outs.iter().filter(|o| !o.is_null() && !plain_tensor(**o)).filter(|o| self.origins.holds(**o, 0)).count() as u32;
            // A result that is alive and no output, and every tuple no output
            // is a part of: its release was not seen.
            let named = |j: u32| outputs.iter().any(|t| matches!(t, Tmpl::Res(s) | Tmpl::Part(s, _) if *s == j));
            let mut late: Vec<u32> = self.origins.alive().into_iter().map(|(j, _)| j).chain(self.tuples.iter().copied()).filter(|j| !named(*j)).collect();
            late.sort();
            self.shadow.late = late.len() as u32;
            self.drops.extend(late);
            self.flush();
            let outside = outer.ops.len().saturating_sub(self.claimed);
            if outer.flags().opaque {
                self.shadow.notes.push(format!("the run's witness could not see: {}", outer.why));
            } else if outside > 0 {
                self.shadow.notes.push(format!("{outside} operator(s) ran between the instructions of the run"));
            }
            let (ninputs, npins) = (self.origins.ninputs, self.origins.pins.len() as u32);
            let consts = self.origins.consts.iter().map(|c| c.clone_ref(Python::assume_attached())).collect();
            match List::new(ninputs, npins, consts, std::mem::take(&mut self.entries), outputs) {
                Err(why) => self.shadow.refused = Some(why),
                Ok(list) => {
                    let py = Python::assume_attached();
                    self.shadow.text = (0..list.entries.len()).map(|k| list.line(py, k)).collect::<Vec<_>>().join("\n");
                    let unsafe_to_run = [(!self.pure, "an entry writes"), (made.is_empty(), "no output of the block is a result of an entry"), (outer.flags().opaque || outside > 0, "operators ran that are under no entry, or could not be seen"), (Origins::lost(), "a death was not noted")];
                    self.shadow.validated = match unsafe_to_run.iter().find(|(so, _)| *so) {
                        Some((_, why)) => format!("not validated: {why}"),
                        None => shadow::validate(py, &list, &self.origins.inputs, &self.origins.pins, &made),
                    };
                }
            }
            self.shadow
        }
    }
}

fn with<R: Default>(f: impl FnOnce(&mut Rec) -> R) -> R {
    REC.with_borrow_mut(|r| r.as_mut().map_or_else(R::default, f))
}

// A run of the block `place` (key: its plan and block) begins on `params`
// in the frame `frame`.
pub unsafe fn begin(place: String, key: (usize, u32), params: &[*mut PyObject], frame: *mut Frame) {
    unsafe {
        TAKEN.with_borrow_mut(|t| t.insert(key));
        let mut origins = Origins::begin();
        params.iter().for_each(|p| origins.input(*p));
        let locals = tr_frame_localsplus(frame);
        (0..tr_frame_nlocalsplus(frame) as usize).for_each(|i| origins.input(*locals.add(i)));
        let shadow = Shadow { place, key, ..Shadow::default() };
        witness::begin();
        REC.set(Some(Rec { shadow, origins, entries: Vec::new(), stack: Vec::new(), window: None, drops: Vec::new(), tuples: Vec::new(), at: 0, claimed: 0, pure: true, left: HashMap::new(), handed: HashMap::new() }));
        LIVE.set(true);
    }
}

pub unsafe fn end(outs: Option<&[*mut PyObject]>) {
    unsafe {
        LIVE.set(false);
        let outer = witness::end();
        // The block's exception, if it left by one, stays as it is.
        let raised = ffi::PyErr_GetRaisedException();
        if let Some(rec) = REC.take() {
            shadow::keep(rec.end(outs, outer));
        }
        if !raised.is_null() {
            ffi::PyErr_SetRaisedException(raised);
        }
    }
}

// True: after() is to be told how the instruction ended.
pub unsafe fn before(body: bool, v: u32, ins: &Instr, site: &Site, callee: Option<*mut PyObject>) -> bool {
    with(|r| unsafe { r.before(body, v, ins, site, callee) })
}

pub unsafe fn after(ok: bool, results: &[*mut PyObject]) {
    with(|r| unsafe { r.after(ok, results) })
}

// An instruction of the executor's inline path; `stores`: the value it
// stores into a local, if it does.
pub fn tick(stores: Option<u32>) {
    with(|r| {
        r.settle();
        r.shadow.seen += 1;
        r.shadow.left_behind += r.shadow.stop.is_some() as u64;
        r.origins.used(DEPTH.get(), stores.into_iter());
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
    with(|r| unsafe {
        let name = || text(tr_frame_qualname(frame));
        if !runs {
            r.stop(&name(), "a frame the executor does not run: nothing of it is seen".into());
        } else if let Some((Pending::Entry(call), ..)) = r.stack.last() {
            let label = call.label.clone();
            r.stop(&label, format!("the Python frame {} ran under the C call of an entry", name()));
        }
    })
}

// Today's engine traces the place with this key.
pub fn traced(key: (usize, u32)) {
    shadow::traced(key);
}
