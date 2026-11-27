// The list of a recorded run, run again on LATER arrivals at its place:
// the validation on other operands that the program itself gives (record.rs
// arms it; run.rs tells when the place is arrived at and what the block made).
//
// What it is given
//   - by record.rs, behind a recording whose list validated equal on the
//     recorded run and whose reads were settled into checks: the list with
//     what it is checked against (Armed), and which of the block's outputs
//     its outputs are (arm);
//   - by run.rs, at a later arrival at the place, before a version of the
//     block is selected: the block's params and frame (begin); when the
//     block has run, however today's engine ran it: its outputs (outs).
// What it guarantees
//   - a list is run again only if record.rs armed it: every entry is one
//     whose row says it only computes, and the witness of none disagreed;
//   - THE CHECKS ARE ASKED FIRST, and a list runs on an arrival only if
//     every one holds (Armed::checks, in the order region.rs's
//     Variant::enter asks a region's; a list will come through that entry
//     as a region whose function is the list). Each "no" is a MISS of the
//     list, counted by the check, and the block runs as written:
//       the frame is of the recorded function and code (so its closure's
//         cells, its globals and its builtins are the recorded ones);
//       the thread is neutral (ambient.rs);
//       the arrival's inputs (origins.rs walk: the params, then the frame's
//         locals, and what the exact tuples, lists and dicts of str keys
//         among them hold, each tensor once) are found by the same paths
//         as the recorded run's, under the same dict keys, with the same
//         keys;
//       an input the list is specialized on (an immutable the recorded run
//         took by value: origins.rs exact) is exactly the recorded value
//         (exec/exact.rs);
//       every root a pin or a stated output was led from is the recorded
//         object at its path; every pin lives and has the key it had; an
//         input that a recorded read also found is the recorded object;
//       the thread's word is the word the recording ran under;
//       what the run read is as it was (reads.rs Reads::check);
//     and, behind every check and whatever a check let go of, the thread's
//     state is read again as the last thing before the list runs;
//   - a list whose function, code, root or pin died, or whose reads expect
//     an object that died, can never pass again: it is dropped where that
//     is found, outside every borrow (what it held may run a finalizer);
//   - what the block made is compared with what the list states: each
//     output that is stated with its statement (outs.rs holds: a value, an
//     object by identity, a new container by its structure), and the
//     tensors in them with what the list made (shadow.rs compare), which is
//     then released. An arrival where a statement cannot be compared is
//     counted "not comparable"; the first difference is kept in the
//     place's Shadow, and the list is not run again after one;
//   - a list that its OWN check ended before the entry the check is for ran,
//     with nothing to take back (run.rs: the stops "class", "part" and
//     "assert" that are restartable, for which no exception is set), is a
//     MISS of the list on that arrival, counted by the check: the block runs
//     as written there, which is what it did. Any other stop (an entry that
//     raised, a stop that is not restartable, a refusal to run) is a
//     difference: the block ran to its end on those operands;
//   - at most LATER arrivals per place: behind the last, and behind the
//     first difference, the list leaves the table. Under `shadow` what its
//     reads count is of the closed immutable list only (finish.rs arms no
//     other list there), so no object of the program's lives longer for a
//     list in the table. Under `1` a list is armed with whatever its reads
//     count (an object whose class takes no weak reference: a namespace, a
//     cell, a container found as a value), which then lives as long as the
//     list is armed: the stated limit of that mode (record.rs).
// What it may not assume
//   - that the block's own outputs are eager's: they are what today's engine
//     made on that arrival (a region of it, or the block as written).

use std::cell::RefCell;
use std::collections::HashMap;

use pyo3::ffi::{self, PyObject};
use pyo3::prelude::*;

use super::origins::{plain_tensor, walk, KEY};
use crate::bindings::ffi::Word;
use crate::exec::ambient;
use crate::exec::ffi::{tr_frame_code, tr_frame_func};
use crate::exec::reads::Reads;
use super::outs::{Kept, Out};
use super::run::{Stop, What};
use super::{shadow, List};
use crate::exec::exact;
use crate::exec::ffi::{tr_frame_localsplus, tr_frame_nlocalsplus, Frame};
use crate::exec::pin::Pin;
use crate::exec::tensor_key::TensorKey;

const LATER: u32 = 16;

pub struct Armed {
    pub list: List,
    pub keys: Vec<TensorKey>,
    // The path each input was found by at the recorded run; the dict keys
    // on those paths; the immutable inputs the list is specialized on, by
    // position.
    pub paths: Vec<Box<[u32]>>,
    pub names: Vec<(Box<[u32]>, Py<PyAny>)>,
    pub exact: Vec<(u32, Py<PyAny>)>,
    pub pins: Vec<Pin>,
    pub pin_keys: Vec<TensorKey>,
    // The roots a pin or a stated output was led from, with their paths.
    pub roots: Vec<(Box<[u32]>, Pin)>,
    // The recorded run's inputs, weakly, and those of them (by number) that
    // a recorded read also found.
    pub inputs: Vec<Pin>,
    pub found_too: Vec<u32>,
    // What the run read, settled: the checks of reads.rs.
    pub reads: Py<Reads>,
    // The thread state the recording ran under.
    pub word: Word,
    // The function and the code of the body's frame.
    pub func: Pin,
    pub code: Pin,
    // The outputs of the block that the list states: the position among
    // the block's outputs, and the statement.
    pub stated: Vec<(usize, Out)>,
    // The place's Shadow (shadow::amend).
    pub kept: usize,
    pub left: u32,
}

struct Pending {
    id: u64,
    key: (u64, u32),
    frame: usize,
    // What the list made (owned), and the storages it ran on.
    made: *mut PyObject,
    operands: Vec<usize>,
    // What the arrival began on at the positions its statements name.
    kept: Vec<(u32, Kept)>,
}

thread_local! {
    static ARMED: RefCell<HashMap<(u64, u32), Armed>> = RefCell::new(HashMap::new());
    static PENDING: RefCell<Vec<Pending>> = const { RefCell::new(Vec::new()) };
    static NEXT: std::cell::Cell<u64> = const { std::cell::Cell::new(1) };
}

pub fn arm(key: (u64, u32), armed: Armed) {
    // The lists that can never pass again leave the table here, with the
    // one this replaces and with the lists of an earlier plan of the same
    // function (a plan made again has a new number; a frame still running
    // the old one loses a comparison and nothing else): the table holds at
    // most one list per block of a function that lives. All are dropped
    // outside the borrow.
    let func = armed.func.as_ptr();
    let gone: Vec<Armed> = ARMED.with_borrow_mut(|a| {
        let dead: Vec<(u64, u32)> = a.iter().filter(|(k, x)| x.gone() || (k.0 < key.0 && x.func.is(func))).map(|(k, _)| *k).collect();
        let mut gone: Vec<Armed> = dead.into_iter().filter_map(|k| a.remove(&k)).collect();
        gone.extend(a.insert(key, Armed { left: LATER, ..armed }));
        gone
    });
    drop(gone);
}

pub fn armed(key: (u64, u32)) -> bool {
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
            // (A list with no arrival left goes with it, outside the borrow.)
            let spent = ARMED.with_borrow_mut(|a| if a.get(&pending.key).is_some_and(|x| x.left == 0) { a.remove(&pending.key) } else { None });
            unsafe { ffi::Py_DECREF(pending.made) };
            drop(spent);
        }
    }
}

// What an arrival is walked into: its inputs as origins.rs numbers them,
// each with its path; the keys of the dicts walked, by path; and what it
// began on (the params, then the frame's locals).
struct Arrival {
    inputs: Vec<*mut PyObject>,
    paths: Vec<Box<[u32]>>,
    keys: HashMap<Box<[u32]>, *mut PyObject>,
    all: Vec<*mut PyObject>,
    // How many of the roots asked for are the recorded object at their path.
    rooted: usize,
}

unsafe fn arrival(params: &[*mut PyObject], frame: *mut Frame, roots: &[(Box<[u32]>, Pin)]) -> Arrival {
    unsafe {
        let locals = tr_frame_localsplus(frame);
        let all: Vec<*mut PyObject> = params.iter().copied().chain((0..tr_frame_nlocalsplus(frame) as usize).map(|i| *locals.add(i))).collect();
        let (mut inputs, mut paths, mut keys): (Vec<*mut PyObject>, Vec<Box<[u32]>>, HashMap<Box<[u32]>, *mut PyObject>) = (Vec::new(), Vec::new(), HashMap::new());
        let mut rooted = 0;
        walk(all.iter().copied(), &mut |path, o| {
            if path.last().is_some_and(|step| step & KEY != 0) {
                keys.insert(path.into(), o);
            } else if plain_tensor(o) {
                if !inputs.contains(&o) {
                    inputs.push(o);
                    paths.push(path.into());
                }
            } else {
                rooted += roots.iter().filter(|(at, root)| **at == *path && root.is(o)).count();
            }
        });
        Arrival { inputs, paths, keys, all, rooted }
    }
}

fn miss(kept: usize, check: &str) {
    shadow::amend(kept, |s| match s.missed.iter_mut().find(|(c, _)| c == check) {
        Some((_, n)) => *n += 1,
        None => s.missed.push((check.to_string(), 1)),
    })
}

// What an arrival that passed the checks runs on: its inputs, the pins
// (counted while the list runs), and what it began on.
type Passed = (Vec<*mut PyObject>, Vec<Py<PyAny>>, Vec<*mut PyObject>);

impl Armed {
    // The checks of a later arrival on `params` (borrowed) in `frame`, in
    // order; Err: the check that does not hold (the miss's reason). Nothing
    // of the list has run.
    unsafe fn checks(&self, params: &[*mut PyObject], frame: *mut Frame) -> Result<Passed, String> {
        unsafe {
            let py = Python::assume_attached();
            if !self.func.is(tr_frame_func(frame)) || !self.code.is(tr_frame_code(frame)) {
                return Err("the function of the body's frame".into());
            }
            let word = ambient::state().map_err(|why| format!("thread state: not neutral: {why}"))?;
            let Arrival { inputs, paths, keys, all, rooted } = arrival(params, frame, &self.roots);
            let named = self.names.iter().all(|(path, name)| keys.get(path).is_some_and(|k| exact::same(*k, name.as_ptr())));
            if !named || paths != self.paths {
                return Err("where the inputs are found".into());
            }
            if inputs.len() != self.keys.len() || !inputs.iter().zip(&self.keys).all(|(o, k)| TensorKey::of(*o).as_ref() == Some(k)) {
                return Err("the key of an input".into());
            }
            // The list is for these values of its immutable inputs.
            if !self.exact.iter().all(|(position, value)| all.get(*position as usize).is_some_and(|o| exact::same(*o, value.as_ptr()))) {
                return Err("input value".into());
            }
            if rooted != self.roots.len() {
                return Err("a root the reads start from".into());
            }
            let pins = self.pins.iter().map(|p| p.live().filter(|o| !o.is_null()).map(|o| Py::<PyAny>::from_borrowed_ptr(py, o))).collect::<Option<Vec<_>>>();
            let Some(pins) = pins.filter(|pins| pins.iter().zip(&self.pin_keys).all(|(o, k)| TensorKey::of(o.as_ptr()).as_ref() == Some(k))) else {
                return Err("a pin".into());
            };
            if !self.found_too.iter().all(|k| self.inputs.get(*k as usize).is_some_and(|recorded| inputs.get(*k as usize).is_some_and(|o| recorded.is(*o)))) {
                return Err("an input a read of the run also found".into());
            }
            // Before anything of the list runs.
            if word != self.word {
                return Err("thread state".into());
            }
            if let Err(read) = self.reads.get().check_for_list() {
                return Err(format!("a read: {read}"));
            }
            // The last look at the thread: a check may have let go of what
            // it held, and what runs for that may have changed the state.
            if ambient::state() != Ok(self.word) {
                return Err("thread state, behind the checks".into());
            }
            Ok((inputs, pins, all))
        }
    }

    // Whether no arrival can pass the checks again.
    fn gone(&self) -> bool {
        self.func.dead() || self.code.dead() || self.roots.iter().any(|(_, root)| root.dead()) || self.pins.iter().any(Pin::dead) || self.reads.get().dead()
    }
}

// A later arrival at the place `key` on `params` (borrowed) in `frame`.
pub unsafe fn begin(key: (u64, u32), params: &[*mut PyObject], frame: *mut Frame) -> Option<Again> {
    unsafe {
        let py = Python::assume_attached();
        // The list leaves the table while it is checked and while it runs:
        // nothing that runs meanwhile can find it.
        let mut armed = ARMED.with_borrow_mut(|a| a.remove(&key))?;
        armed.left -= 1;
        let (inputs, pins, all) = match armed.checks(params, frame) {
            Ok(passed) => passed,
            Err(check) => {
                miss(armed.kept, &check);
                // (Dropped here, where nothing is borrowed.)
                if !armed.gone() && armed.left > 0 {
                    ARMED.with_borrow_mut(|a| a.insert(key, armed));
                }
                return None;
            }
        };
        let pointers: Vec<*mut PyObject> = pins.iter().map(|o| o.as_ptr()).collect();
        let ran = crate::exec::outside_executor(|| armed.list.run(py, &inputs, &pointers));
        let found = match ran {
            Ok(made) => {
                let id = NEXT.replace(NEXT.get() + 1);
                let operands = inputs.iter().chain(&pointers).map(|o| shadow::storage(*o)).collect();
                let mut positions = Vec::new();
                armed.stated.iter().for_each(|(_, out)| out.inputs(&mut positions));
                let kept = positions.into_iter().map(|p| (p, Kept::of(all.get(p as usize).copied().unwrap_or(std::ptr::null_mut())))).collect();
                PENDING.with_borrow_mut(|p| p.push(Pending { id, key, frame: frame as usize, made, operands, kept }));
                Some(Again(id))
            }
            Err(stop) => {
                ffi::PyErr_Clear();
                match stopped(&stop) {
                    Ok(check) => miss(armed.kept, check),
                    Err(why) => {
                        shadow::amend(armed.kept, |s| s.again.2 = Some(why));
                        armed.left = 0;
                    }
                }
                None
            }
        };
        // (A list that stopped for good, or whose block's outputs are not
        // awaited and that has no arrival left, is dropped here.)
        if armed.left > 0 || found.is_some() {
            ARMED.with_borrow_mut(|a| a.insert(key, armed));
        }
        found
    }
}

// The block of the place `key` in `frame` has run and made `outs`
// (borrowed, in the order of the block's outputs). A list that is not to run
// again leaves the table here and is dropped outside the borrow: what its
// reads count is let go of with it.
pub unsafe fn outs(key: (u64, u32), frame: *mut Frame, outs: &[*mut PyObject]) {
    unsafe {
        let found = PENDING.with_borrow_mut(|p| p.iter().rposition(|x| x.key == key && x.frame == frame as usize).map(|i| p.remove(i)));
        let Some(pending) = found else { return };
        if let Some(mut armed) = ARMED.with_borrow_mut(|a| a.remove(&key)) {
            // The block's tensor for each output of the list, found by the
            // statements; a statement that does not hold is the difference.
            let mut wanted = vec![std::ptr::null_mut(); armed.list.outputs.len()];
            let mut compared = Ok(true);
            for (at, out) in &armed.stated {
                compared = match outs.get(*at).map(|o| out.holds(*o, &pending.kept, &mut wanted)) {
                    Some(Ok(all)) => compared.map(|so_far| so_far && all),
                    Some(Err(why)) => Err(format!("DIFFERENT: output {at} of the block is {why}")),
                    None => compared.map(|_| false),
                };
                if compared.is_err() {
                    break;
                }
            }
            let verdict = match compared {
                Err(why) => why,
                Ok(all) if !all || wanted.iter().any(|o| o.is_null()) => "not validated".to_string(),
                Ok(_) => shadow::compare(pending.made, &wanted, &pending.operands),
            };
            if verdict == "equal" {
                shadow::amend(armed.kept, |s| s.again.0 += 1);
            } else if verdict.starts_with("DIFFERENT") {
                shadow::amend(armed.kept, |s| s.again.2 = Some(verdict.replacen("DIFFERENT", "DIFFERENT on a later arrival", 1)));
                armed.left = 0;
            } else {
                shadow::amend(armed.kept, |s| s.again.1 += 1);
            }
            if armed.left > 0 {
                ARMED.with_borrow_mut(|a| a.insert(key, armed));
            }
        }
        ffi::Py_DECREF(pending.made);
    }
}
