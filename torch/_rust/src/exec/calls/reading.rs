// The reads of a recorded run (record.rs asks; nothing else does): when the
// recorder has the thread's reads (reads.rs Owner), and when it lets go.
//
// What it is given: the moments of a recording: its begin, a stop, the next
// thing the recorder is told behind a stop, its end.
// What it guarantees
//   - begin() is Ok only under a neutral ambient state (ambient.rs), whose
//     word it answers, and with the reads of the thread the recorder's from
//     there on. What the hooks
//     of a reading run look up once per process (reads.rs, writes.rs:
//     Python snippets) is looked up here, outside the executor and before
//     the body, never under an instruction of the run;
//   - the reads are let go where the recording ends (end) or, behind a stop,
//     where the recorder is next told something (stopped, then release):
//     never by the caller of stopped(), which holds the recordings borrowed,
//     because what the reads held is released then and a finalizer may run;
//   - end() says, before it lets go, whether every read stood (reads.rs
//     standing); if so it settles them as a trace's reads are settled
//     (reads.rs settle_for_recorder) and answers what a list of the run is
//     checked against on a later arrival, or why the reads are no check.
// What it may not assume: that reads which began still stand; record.rs asks
// standing() before every instruction.

use std::cell::Cell;

use pyo3::ffi::PyObject;
use pyo3::prelude::*;

use crate::bindings::ffi::Word;
use crate::exec::{ambient, backward, outside_executor, reads, rules, writes};

thread_local! {
    // A recording that stopped has reads to let go of.
    static RELEASE: Cell<bool> = const { Cell::new(false) };
}

pub fn begin() -> Result<Word, String> {
    release();
    let word = ambient::state().map_err(|what| format!("the state around the run is not neutral: {what}"))?;
    let began = outside_executor(|| {
        // (run.rs asks of every call whether it starts the autograd engine.)
        backward::ready();
        let began = reads::begin_for_recorder();
        if began {
            writes::begin();
            // (Made where a function goes onto the executor; asked for here
            // so that rules::license never makes it under an instruction.)
            rules::ready();
        }
        began
    });
    if began { Ok(word) } else { Err("the reads of this thread are a trace's: the run's own cannot be recorded".into()) }
}

pub fn stopped() {
    RELEASE.set(true);
}

pub fn release() {
    if RELEASE.replace(false) {
        reads::end_for_recorder();
    }
}

// What the end of a run's reads is: every read was recorded and they are
// the checks of a list (Checks), or they are no check of a later arrival
// (Loose: why), or not every read of the run was recorded (Fell: why).
pub enum Ended {
    Checks(Py<reads::Reads>),
    Loose(String),
    Fell(String),
}

// The run is over: `ours`, its inputs and the results of its entries that
// live; `cells`, the cells its body's frame made for the call.
pub unsafe fn end(ours: &[*mut PyObject], cells: &[*mut PyObject]) -> Ended {
    if let Err(why) = reads::standing() {
        reads::end_for_recorder();
        return Ended::Fell(why);
    }
    match unsafe { outside_executor(|| reads::settle_for_recorder(ours, cells)) } {
        Ok(checks) => Ended::Checks(checks),
        Err(why) => Ended::Loose(why),
    }
}
