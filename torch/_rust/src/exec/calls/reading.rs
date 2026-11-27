// The reads of a recorded run (record.rs asks; nothing else does): when the
// recorder has the thread's reads (reads.rs Owner), and when it lets go.
//
// What it is given: the moments of a recording: its begin, a stop, the next
// thing the recorder is told behind a stop, its end.
// What it guarantees
//   - begin() is Ok only under a neutral ambient state (ambient.rs) and with
//     the reads of the thread the recorder's from there on. What the hooks
//     of a reading run look up once per process (reads.rs, writes.rs:
//     Python snippets) is looked up here, outside the executor and before
//     the body, never under an instruction of the run;
//   - the reads are let go where the recording ends (end) or, behind a stop,
//     where the recorder is next told something (stopped, then release):
//     never by the caller of stopped(), which holds the recordings borrowed,
//     because what the reads held is released then and a finalizer may run;
//   - end() says, before it lets go, whether every read stood (reads.rs
//     standing).
// What it may not assume: that reads which began still stand; record.rs asks
// standing() before every instruction.

use std::cell::Cell;

use crate::exec::{ambient, backward, outside_executor, reads, rules, writes};

thread_local! {
    // A recording that stopped has reads to let go of.
    static RELEASE: Cell<bool> = const { Cell::new(false) };
}

pub fn begin() -> Result<(), String> {
    release();
    if let Some(what) = ambient::not_neutral() {
        return Err(format!("the state around the run is not neutral: {what}"));
    }
    let began = outside_executor(|| {
        // (run.rs asks of every call whether it starts the autograd engine.)
        backward::ready();
        let began = reads::begin_for_recorder();
        if began {
            writes::begin();
            // (Made where a function goes onto the executor; asked for here
            // so that rules::license never makes it under an instruction.)
            rules::torch::ready();
        }
        began
    });
    if began { Ok(()) } else { Err("the reads of this thread are a trace's: the run's own cannot be recorded".into()) }
}

pub fn stopped() {
    RELEASE.set(true);
}

pub fn release() {
    if RELEASE.replace(false) {
        reads::end_for_recorder();
    }
}

// Why not every read of the run was recorded, if that is so.
pub fn end() -> Option<String> {
    let stood = reads::standing().err();
    reads::end_for_recorder();
    stood
}
