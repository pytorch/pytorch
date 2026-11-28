// Code the executor leaves to CPython for good, and why: the "on CPython"
// rows of torch.vm.regions.
//
// The executor runs a frame on a plan made from the frame's code. Where a
// plan cannot be made, the outcome is one, whatever failed: the frames of
// that code run on CPython, as written, from then on (the frames they
// enter are the executor's again), and the reason is a row of the report.
// Never an exception into the program, never a traceback on the side.
//   - no first plan (the code has something the executor does not run, or
//     the plan built from it does not verify), or nowhere to keep one with
//     the code: `code`, from the hook's entry, which marks the code so that
//     it is not tried again;
//   - no later plan (what the optimizer made of a code that ran does not
//     verify, or could not be made): `compiled`, from wherever a code is
//     planned again without the program asking.
// That is for an error that says something of the code. One that does not
// (`of_the_code`: an interrupt, an exit, memory that ran out) makes no row
// and leaves the code as it was: it goes where it went before there were
// rows (raised into the program from a first plan, to sys.unraisablehook
// from a plan made after a frame returned, dropped by a planning round).
// Frames that arrive while a tool is set through sys.settrace,
// sys.setprofile or sys.monitoring are CPython's too, for as long as it is
// set and not for good: one row counts them (`tool`).
// A region that was traced and that the plan of its code cannot take is no
// plan failure: the code runs on as it did, and the trace says why it is no
// region (exec::add_region).
// At most MAX_ROWS rows are kept, each (name, line, reason) once; a code
// past that bound runs on CPython all the same, without a row.

use std::os::raw::c_void;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use pyo3::prelude::*;

use super::Compiled;

const MAX_ROWS: usize = 4096;

// (qualname, first line, why), each once.
static ROWS: Mutex<Vec<(String, i32, String)>> = Mutex::new(Vec::new());

// co_extra value of code with no first plan; the hook passes its frames to
// CPython's evaluator.
static MARK: u8 = 0;

pub fn mark() -> *mut c_void {
    &MARK as *const u8 as *mut c_void
}

fn row(name: String, line: i32, why: String) {
    let mut rows = ROWS.lock().unwrap();
    if rows.len() < MAX_ROWS && !rows.iter().any(|r| r.0 == name && r.1 == line && r.2 == why) {
        rows.push((name, line, why));
    }
}

// Whether an error that came while a plan was made says something of the
// code: an Exception that is no MemoryError. (KeyboardInterrupt and
// SystemExit are no Exception; the case nobody listed is of the code.)
pub fn of_the_code(py: Python<'_>, e: &PyErr) -> bool {
    e.is_instance_of::<pyo3::exceptions::PyException>(py) && !e.is_instance_of::<pyo3::exceptions::PyMemoryError>(py)
}

// No first plan for the code named `name`, which starts at `line`: none
// could be made (`making`), or it could not be kept with the code.
pub fn code(name: String, line: i32, making: bool, why: String) {
    let what = if making { "no plan could be made for its code" } else { "no plan can be kept with its code" };
    row(name, line, format!("{what}: {why}"));
}

// A later plan of `c` could not be made, for `e`. If `e` is of the code,
// its frames run on CPython from now on (under a traced body the executor
// still runs them, on the plan it has, which verified); else `e` is handed
// back and nothing changes.
pub fn compiled(py: Python<'_>, c: &Compiled, e: PyErr) -> Option<PyErr> {
    if !of_the_code(py, &e) {
        return Some(e);
    }
    c.cold.store(true, Ordering::Relaxed);
    c.deep.store(false, Ordering::Relaxed);
    row(c.qualname.clone(), c.lines.first().map_or(0, |l| l.2), format!("its code could not be planned again: {e}"));
    None
}

// How many frames arrived while a tool was set.
static TOOL_FRAMES: AtomicUsize = AtomicUsize::new(0);

// A frame arrived while a tool was set: CPython runs it.
pub fn tool() {
    TOOL_FRAMES.fetch_add(1, Ordering::Relaxed);
}

/// What the executor leaves to CPython, as (qualname, first line, why): the
/// codes no plan could be made for, and, under the name "*", the frames
/// that arrived while a tool was set through sys.settrace, sys.setprofile
/// or sys.monitoring.
#[pyfunction]
pub fn _cfg_left() -> Vec<(String, i32, String)> {
    let mut rows = ROWS.lock().unwrap().clone();
    let n = TOOL_FRAMES.load(Ordering::Relaxed);
    if n > 0 {
        rows.push(("*".into(), 0, format!("{n} frames arrived while a tool was set through sys.settrace, sys.setprofile or sys.monitoring: CPython ran them, so that the tool was told of them")));
    }
    rows
}

// For tests of the outcome: plans of code with this qualname are taken not
// to verify (or, with the last flag, to meet a KeyboardInterrupt); `true`:
// the first plan, else every later one. FAIL_ON: whether there is one (all
// that a plan reads of this when no test asked).
static FAIL: Mutex<Option<(String, bool, bool)>> = Mutex::new(None);
static FAIL_ON: AtomicBool = AtomicBool::new(false);

/// For tests: take the plans of the code named `qualname` not to verify (the
/// first plan with `first`, else the plans made of it after it ran), or,
/// with `interrupt`, to be interrupted (a KeyboardInterrupt); None ends it.
#[pyfunction]
#[pyo3(signature = (qualname, first=false, interrupt=false))]
pub fn _cfg_fail_plans(qualname: Option<String>, first: bool, interrupt: bool) {
    let mut fail = FAIL.lock().unwrap();
    FAIL_ON.store(qualname.is_some(), Ordering::Relaxed);
    *fail = qualname.map(|q| (q, first, interrupt));
}

// Err if a test asked for the plans of the code that `qualname` names to
// fail (see above). The name is asked for only if a test asked.
pub fn asked_to_fail(first: bool, qualname: impl FnOnce() -> PyResult<String>) -> PyResult<()> {
    if !FAIL_ON.load(Ordering::Relaxed) {
        return Ok(());
    }
    let name = qualname()?;
    match &*FAIL.lock().unwrap() {
        Some((q, f, true)) if *q == name && *f == first => Err(pyo3::exceptions::PyKeyboardInterrupt::new_err("a test asked for it")),
        Some((q, f, false)) if *q == name && *f == first => Err(pyo3::exceptions::PyRuntimeError::new_err("CFG verification failed: a test asked for it")),
        _ => Ok(()),
    }
}
