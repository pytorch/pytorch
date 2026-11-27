// Which suspended generator frames are the executor's to resume.
//
// THE INVARIANT: the executor runs a frame only from its first instruction,
// or from a place where the executor itself suspended it and nothing else
// has run it since. Every other frame that arrives at the eval-frame hook
// is CPython's, untouched: a generator, coroutine or async generator that
// CPython started or last suspended, whatever resumes it (next, send,
// throw, close, the collector, the interpreter's shutdown) and from
// wherever (the body of a function the executor runs, code outside it, a
// finalizer).
//
// How the executor knows: by a mark of its own, never by what the frame
// looks like. A frame is in OWN from the moment the executor suspends it
// (exec::leave) until it next arrives at the hook, where the mark is taken
// whoever runs the frame then: the executor, or CPython for any of the
// hook's reasons (no executor on the thread, cold code, the tracer's code),
// after which the frame is CPython's for good. While the hook is not
// installed CPython resumes frames unseen, so the marks do not outlive it:
// uninstalling forgets them all (exec::live), and a thread that steps
// outside the executor for a while (exec::outside_executor) leaves the hook
// installed if there are any. So a mark that is found says
// that the last one to run the frame was the executor: every resume since
// came through the hook and would have taken it. That also covers a frame
// that died with its mark (a generator freed with no resume): whatever is
// allocated at its address is new, its first resume comes through the hook
// with nothing of it run, where the mark is taken and not asked for.
//
// A frame nothing has run (CPython made it; its first resume) is the
// executor's to start when a value is sent, and CPython's when an exception
// is thrown into it: nothing of it runs then.

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use super::ffi::Frame;

// By address, only touched with the GIL held. A frame's address is its
// generator object's, which is not reused while the generator lives.
static OWN: Mutex<Option<HashSet<usize>>> = Mutex::new(None);
// How many marks there are: most frames at the hook find none.
static MARKS: AtomicUsize = AtomicUsize::new(0);

// The executor has suspended `frame`.
pub fn suspended(frame: *mut Frame) {
    let mut own = OWN.lock().unwrap();
    own.get_or_insert_with(HashSet::new).insert(frame as usize);
    MARKS.store(own.as_ref().map_or(0, |s| s.len()), Ordering::Relaxed);
}

// Whether the executor suspended `frame` and nothing has resumed it since;
// it is being resumed now, so the mark goes.
pub fn take(frame: *mut Frame) -> bool {
    if MARKS.load(Ordering::Relaxed) == 0 {
        return false;
    }
    let mut own = OWN.lock().unwrap();
    let was = own.as_mut().is_some_and(|s| s.remove(&(frame as usize)));
    MARKS.store(own.as_ref().map_or(0, |s| s.len()), Ordering::Relaxed);
    was
}

// Whether the executor has suspended frames to resume.
pub fn any() -> bool {
    MARKS.load(Ordering::Relaxed) != 0
}

// The hook is being uninstalled: resumes are not seen from now on.
pub fn forget_all() {
    if MARKS.load(Ordering::Relaxed) != 0 {
        *OWN.lock().unwrap() = None;
        MARKS.store(0, Ordering::Relaxed);
    }
}

/// How many suspended generator frames the executor would resume itself
/// (for tests).
#[pyo3::pyfunction]
pub fn _cfg_own_frames() -> usize {
    MARKS.load(Ordering::Relaxed)
}
