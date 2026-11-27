// The list of a recorded run, run again on LATER arrivals at its place:
// the validation on other operands that the program itself gives (record.rs
// arms it; run.rs tells when the place is arrived at and what the block made).
//
// What it is given
//   - by record.rs, behind a recording whose list validated equal on the
//     recorded run: the list, the keys of its inputs, its pins held weakly,
//     which of the block's outputs its outputs are (arm);
//   - by run.rs, at a later arrival at the place, before a version of the
//     block is selected: the block's params and frame (begin); when the
//     block has run, however today's engine ran it: its outputs (outs).
// What it guarantees
//   - a list is run again only if record.rs armed it: every entry is one
//     whose row says it only computes, and the witness of none disagreed;
//   - it runs on an arrival only if that arrival's inputs (origins.rs walk:
//     the params, then the frame's locals, and the items of the exact tuples
//     and lists among them, each tensor once) are found by the same paths as
//     the recorded run's and have the same keys, and every pin is alive; any
//     other arrival is counted "not comparable";
//   - what the list made is compared with what the block made (shadow.rs
//     compare) and then released; the first difference is kept in the
//     place's Shadow, and the list is not run again after one;
//   - a list that its OWN check ended before the entry the check is for ran,
//     with nothing to take back (run.rs: the stops "class", "part" and
//     "assert" that are restartable, for which no exception is set), is a
//     MISS of the list on that arrival, counted by the check: the block runs
//     as written there, which is what it did. Any other stop (an entry that
//     raised, a stop that is not restartable, a refusal to run) is a
//     difference: the block ran to its end on those operands;
//   - at most LATER arrivals per place.
// What it may not assume
//   - that the block's own outputs are eager's: they are what today's engine
//     made on that arrival (a region of it, or the block as written).

use std::cell::RefCell;
use std::collections::HashMap;

use pyo3::ffi::{self, PyObject};
use pyo3::prelude::*;

use super::origins::{plain_tensor, walk};
use super::run::{Stop, What};
use super::{shadow, List};
use crate::exec::ffi::{tr_frame_localsplus, tr_frame_nlocalsplus, Frame};
use crate::exec::pin::Pin;
use crate::exec::tensor_key::TensorKey;

const LATER: u32 = 16;

pub struct Armed {
    pub list: List,
    pub keys: Vec<TensorKey>,
    // The path each input was found by at the recorded run.
    pub paths: Vec<Box<[u32]>>,
    pub pins: Vec<Pin>,
    // For each output of the list, the position among the block's outputs.
    pub outs: Vec<usize>,
    // The place's Shadow (shadow::amend).
    pub kept: usize,
    pub left: u32,
}

struct Pending {
    id: u64,
    key: (usize, u32),
    frame: usize,
    // What the list made (owned), and the storages it ran on.
    made: *mut PyObject,
    operands: Vec<usize>,
}

thread_local! {
    static ARMED: RefCell<HashMap<(usize, u32), Armed>> = RefCell::new(HashMap::new());
    static PENDING: RefCell<Vec<Pending>> = const { RefCell::new(Vec::new()) };
    static NEXT: std::cell::Cell<u64> = const { std::cell::Cell::new(1) };
}

pub fn arm(key: (usize, u32), armed: Armed) {
    ARMED.with_borrow_mut(|a| a.insert(key, Armed { left: LATER, ..armed }));
}

pub fn armed(key: (usize, u32)) -> bool {
    ARMED.with_borrow(|a| a.get(&key).is_some_and(|x| x.left > 0))
}

// What a stop of a list on a later arrival is. Ok(the check): a miss, only
// where the list's interpreter itself says that nothing is to be taken back
// (Stop::restartable: every entry up to the stop is reversible, run.rs and
// check.rs) AND the stop is one of the list's own checks. Err(the text): a
// difference, for every other stop. Nothing here works out whether a stop
// can be taken back.
pub fn stopped(stop: &Stop) -> Result<&'static str, String> {
    match stop.what {
        What::Class | What::Part | What::Assert if stop.restartable => Ok(stop.what.text()),
        _ => Err(format!("DIFFERENT on a later arrival: the list stopped at entry {} ({})", stop.entry, stop.what.text())),
    }
}

// An arrival whose block's outputs are awaited. Dropped with none told: the
// block did not reach its end, and what the list made is released.
pub struct Again(u64);

impl Drop for Again {
    fn drop(&mut self) {
        let gone = PENDING.with_borrow_mut(|p| p.iter().position(|x| x.id == self.0).map(|i| p.remove(i)));
        if let Some(pending) = gone {
            unsafe { ffi::Py_DECREF(pending.made) };
        }
    }
}

// The inputs of an arrival as origins.rs numbers them, each with its path.
unsafe fn inputs(params: &[*mut PyObject], frame: *mut Frame) -> (Vec<*mut PyObject>, Vec<Box<[u32]>>) {
    unsafe {
        let locals = tr_frame_localsplus(frame);
        let all = params.iter().copied().chain((0..tr_frame_nlocalsplus(frame) as usize).map(|i| *locals.add(i)));
        let (mut found, mut paths): (Vec<*mut PyObject>, Vec<Box<[u32]>>) = (Vec::new(), Vec::new());
        walk(all, &mut |path, o| {
            if plain_tensor(o) && !found.contains(&o) {
                found.push(o);
                paths.push(path.into());
            }
        });
        (found, paths)
    }
}

// A later arrival at the place `key` on `params` (borrowed) in `frame`.
pub unsafe fn begin(key: (usize, u32), params: &[*mut PyObject], frame: *mut Frame) -> Option<Again> {
    unsafe {
        let py = Python::assume_attached();
        let (inputs, paths) = inputs(params, frame);
        // The list leaves the table while it runs: nothing it runs can find it.
        let mut armed = ARMED.with_borrow_mut(|a| a.remove(&key))?;
        armed.left -= 1;
        let same = paths == armed.paths && inputs.len() == armed.keys.len() && inputs.iter().zip(&armed.keys).all(|(o, k)| TensorKey::of(*o).as_ref() == Some(k));
        let pins = armed.pins.iter().map(|p| p.live().filter(|o| !o.is_null()).map(|o| Py::<PyAny>::from_borrowed_ptr(py, o))).collect::<Option<Vec<_>>>();
        let (true, Some(pins)) = (same, pins) else {
            shadow::amend(armed.kept, |s| s.again.1 += 1);
            ARMED.with_borrow_mut(|a| a.insert(key, armed));
            return None;
        };
        let pointers: Vec<*mut PyObject> = pins.iter().map(|o| o.as_ptr()).collect();
        let ran = crate::exec::outside_executor(|| armed.list.run(py, &inputs, &pointers));
        let found = match ran {
            Ok(made) => {
                let id = NEXT.replace(NEXT.get() + 1);
                let operands = inputs.iter().chain(&pointers).map(|o| shadow::storage(*o)).collect();
                PENDING.with_borrow_mut(|p| p.push(Pending { id, key, frame: frame as usize, made, operands }));
                Some(Again(id))
            }
            Err(stop) => {
                ffi::PyErr_Clear();
                match stopped(&stop) {
                    Ok(check) => shadow::amend(armed.kept, |s| match s.missed.iter_mut().find(|(c, _)| c == check) {
                        Some((_, n)) => *n += 1,
                        None => s.missed.push((check.to_string(), 1)),
                    }),
                    Err(why) => {
                        shadow::amend(armed.kept, |s| s.again.2 = Some(why));
                        armed.left = 0;
                    }
                }
                None
            }
        };
        ARMED.with_borrow_mut(|a| a.insert(key, armed));
        found
    }
}

// The block of the place `key` in `frame` has run and made `outs`
// (borrowed, in the order of the block's outputs).
pub unsafe fn outs(key: (usize, u32), frame: *mut Frame, outs: &[*mut PyObject]) {
    unsafe {
        let found = PENDING.with_borrow_mut(|p| p.iter().rposition(|x| x.key == key && x.frame == frame as usize).map(|i| p.remove(i)));
        let Some(pending) = found else { return };
        ARMED.with_borrow_mut(|a| {
            let Some(armed) = a.get_mut(&key) else { return };
            let wanted: Option<Vec<*mut PyObject>> = armed.outs.iter().map(|i| outs.get(*i).copied().filter(|o| !o.is_null())).collect();
            let verdict = wanted.map_or("not validated".to_string(), |wanted| shadow::compare(pending.made, &wanted, &pending.operands));
            if verdict == "equal" {
                shadow::amend(armed.kept, |s| s.again.0 += 1);
            } else if verdict.starts_with("DIFFERENT") {
                shadow::amend(armed.kept, |s| s.again.2 = Some(verdict.replacen("DIFFERENT", "DIFFERENT on a later arrival", 1)));
                armed.left = 0;
            } else {
                shadow::amend(armed.kept, |s| s.again.1 += 1);
            }
        });
        ffi::Py_DECREF(pending.made);
    }
}
