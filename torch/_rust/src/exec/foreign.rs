// A frame under a trace that no instruction of the traced body called.
//
// A region is made of what the instructions of the traced body did: every
// frame an instruction enters runs on the executor and is recorded (reads.rs,
// writes.rs). Code that runs under the trace because an object DIED is
// called by no instruction: a __del__, the callback of a weak reference,
// weakref.finalize, a generator closed where it was suspended. The object
// dies where its last reference goes: when the executor drops a value
// between two instructions, when a callee's frame ends, or when the trace
// lets go of something it held itself (what a store replaced, the operands
// of a write, an entry of a read, a copy the tracer took before the body).
// What that code read is no entry and what it changed is no write, so a
// region made of the trace would leave it out on every run.
//
// What this module is given
//   - by the eval-frame hook (mod.rs): every frame entered while `judged`
//     says so: (a) while the traced body runs and no instruction of it is in
//     flight, and (b) while a `Releasing` lives;
//   - by the trace's own Rust code: a `Releasing` around every place where
//     it drops references it counted (reads.rs, writes.rs, region.rs);
//   - by the tracer's Python code, before the body runs: the function
//     objects of its own weak-reference callbacks (`_region_callbacks`), and
//     whatever of the program's it holds itself (`_region_keep`);
//   - by region.rs and writes.rs: what they hand to the tracer's Python code
//     (the body's outputs, the operands of its recorded writes): `keep`.
// What it guarantees
//   - `gate()` is Some(why) if one such frame was entered since `begin()`
//     that is none of the named callbacks (compared by identity of the
//     function object, never by a name), nor the executor's own empty
//     function that lets CPython handle what is pending (trampoline.rs), nor
//     the collector's or the interpreter's own callbacks (mod.rs): the trace
//     is then no region. It is asked once, by region::trace, behind every
//     release, and nowhere else;
//   - what was kept is counted until the gate and let go of there, under a
//     `Releasing`: no object the tracer's Python code holds through it can
//     die while that code runs, where a frame run by a death cannot be told
//     from a frame that code calls;
//   - the gate is also where the trace lets go of the records it parked
//     (parked.rs: everything writes.rs and reads.rs would have dropped
//     between the body's last instruction and here, while the tracer's
//     modes were on or its stand-ins in place), the first parked first, and
//     then of what was kept: a finalizer that runs for either runs as code
//     of the program's in the program's own state, and is seen.
// What it may not assume, and does not see
//   - C code run by a death (a C callable as the callback of a weak
//     reference, a C type's deallocator): no frame is entered;
//   - a frame run by a death inside an instruction in flight (it is recorded
//     as that instruction's, like a frame the instruction calls), or inside
//     the tracer's own Python code (handlers under an instruction, the code
//     after the body) for an object that was not kept;
//   - an object the region itself keeps alive: its finalizer never runs.

use std::cell::{Cell, RefCell};

use pyo3::ffi::PyObject;
use pyo3::prelude::*;

use super::ffi::*;

thread_local! {
    // How many `Releasing` live on this thread.
    static RELEASING: Cell<u32> = const { Cell::new(0) };
    // One of the trace's own callbacks is running: what it calls is its own.
    static OWN: Cell<bool> = const { Cell::new(false) };
    // The first frame that was not the body's, as the refusal says it.
    static FOUND: RefCell<Option<String>> = const { RefCell::new(None) };
    // The trace's own callbacks, by function object.
    static CALLBACKS: RefCell<Vec<Py<PyAny>>> = const { RefCell::new(Vec::new()) };
    // What the tracer's Python code holds of the program's, until the gate.
    static KEPT: RefCell<Vec<Py<PyAny>>> = const { RefCell::new(Vec::new()) };
    // How many `Closing` live on this thread.
    static CLOSING: Cell<u32> = const { Cell::new(0) };
}

// Lives while a trace is traced and closed: from its begin until the
// executor has read everything the trace left for it (the gate's answer,
// whether it was refused, the error of a write that could not be taken
// back or of a stand-in that stayed: region::trace hands the guard to its
// caller). And while a gate drains, whoever asked it (the call-list
// recorder asks one too). NO TRACE BEGINS WHILE ONE LIVES, and no region is
// entered: what a death at the gate runs is code of the program's, which
// may call a function torch.vm compiled, and a trace begun there would
// reset what the trace being closed has not handed over yet (`begin` here,
// reads::begin, writes::begin, region::trace's own cells). Such a call runs
// as written (run.rs asks region::engaged at every place that begins a
// trace or enters a region).
pub struct Closing(());

impl Closing {
    pub fn start() -> Closing {
        CLOSING.set(CLOSING.get() + 1);
        Closing(())
    }
}

impl Drop for Closing {
    fn drop(&mut self) {
        CLOSING.set(CLOSING.get() - 1);
    }
}

// Whether a trace is being closed on this thread (or traced: see Closing).
#[inline]
pub fn closing() -> bool {
    CLOSING.get() > 0
}

// A trace begins on this thread.
pub fn begin() {
    let _releasing = Releasing::start();
    FOUND.set(None);
    drop(CALLBACKS.take());
    super::parked::begin();
    drop(KEPT.take());
}

// Lives where the trace's own code drops references it counted: every frame
// entered meanwhile was run by a death. (The hook is installed for as long,
// whether or not an executor is active on the thread.)
pub struct Releasing(#[allow(dead_code)] super::FramesHooked);

impl Releasing {
    pub fn start() -> Releasing {
        RELEASING.set(RELEASING.get() + 1);
        Releasing(super::FramesHooked::new())
    }
}

impl Drop for Releasing {
    fn drop(&mut self) {
        RELEASING.set(RELEASING.get() - 1);
    }
}

// Lives while one of the trace's own callbacks runs.
pub struct Own(bool);

impl Drop for Own {
    fn drop(&mut self) {
        OWN.set(self.0);
    }
}

// Whether a frame entered now is one no instruction of the body called.
// `active`: an executor is active on this thread.
#[inline]
pub fn judged(active: bool) -> bool {
    !OWN.get() && (RELEASING.get() > 0 || (active && super::region::tracing() && super::reads::on() && !super::reads::in_instr()))
}

// A frame was entered while `judged`. Some: it is one of the trace's own
// callbacks, and no frame is judged while the guard lives (what it calls is
// its own). None: unless the collector or the interpreter runs it (mod.rs),
// the trace is no region.
pub unsafe fn entered(frame: *mut Frame) -> Option<Own> {
    let func = tr_frame_func(frame);
    if CALLBACKS.with_borrow(|c| c.iter().any(|f| f.as_ptr() == func)) {
        return Some(Own(OWN.replace(true)));
    }
    // The executor's own empty function, by which CPython's loop handles
    // what is pending between two instructions (trampoline.rs): no code of
    // the program's. What runs UNDER it (a signal's handler, the finalizer
    // of garbage the collector found) is judged like any frame.
    if super::trampoline::is_breaker(func) {
        return None;
    }
    // (As mod.rs asks it: not of a generator's frame, which may be entered
    // with an exception pending.)
    if tr_frame_owned_by_generator(frame) == 0 && (super::gc_callback(frame) || tr_event_hook(func) != 0) {
        return None;
    }
    FOUND.with_borrow_mut(|found| {
        if found.is_none() {
            let when = if RELEASING.get() > 0 { "when the trace let go of what it held" } else { "between two instructions" };
            // Which code it is: its name, the module of its globals, its
            // line. A generator that is being closed is entered with an
            // exception pending: nothing here may set or clear one.
            let mut of = String::from("?");
            if pyo3::ffi::PyErr_Occurred().is_null() {
                let key = pyo3::ffi::PyUnicode_InternFromString(c"__name__".as_ptr());
                let module = if key.is_null() { std::ptr::null_mut() } else { super::reads::dict_get(tr_frame_globals(frame), key) };
                if !module.is_null() {
                    of = super::reads::text(module);
                }
                pyo3::ffi::Py_XDECREF(key);
                pyo3::ffi::PyErr_Clear();
            }
            let line = PyCode_Addr2Line(tr_frame_code(frame), 0);
            *found = Some(format!("a frame that is not the body's ran under it ({} of {of}, line {line}, {when})", super::reads::text(tr_frame_qualname(frame))));
        }
    });
    None
}

// Counts `o` (borrowed) until the gate.
pub unsafe fn keep(o: *mut PyObject) {
    if !o.is_null() {
        KEPT.with_borrow_mut(|k| k.push(super::reads::own(o)));
    }
}

// The one gate of a trace: lets go of what was parked and of what was kept,
// and says why the trace is no region if a frame that was not the body's ran
// under it. Nothing a region's trace holds is released behind it: what the
// region keeps, it keeps. (A trace that is no region lets go of the rest
// right behind it, in region::trace: there is no region to leave anything
// out of, and the thread is in the program's own state.)
pub fn gate() -> Option<String> {
    // (What the drain runs begins no trace: nothing resets FOUND, or what
    // the caller reads behind this, before it is read.)
    let _closing = Closing::start();
    {
        let _releasing = Releasing::start();
        // What the trace would have dropped while its tracer's modes were
        // on, in that order (what its stores replaced first, in the order
        // of the stores): before what the tracer's code was handed. An
        // object the trace judged dead that something refers to again is
        // one its judgments were wrong about: no region.
        if !super::parked::drain() {
            FOUND.with_borrow_mut(|found| {
                found.get_or_insert_with(|| "an object its trace took for one nothing else refers to was referred to again before the trace was over".into());
            });
        }
        // One at a time, the borrow given back first: a finalizer may run.
        while let Some(o) = KEPT.with_borrow_mut(Vec::pop) {
            drop(o);
        }
        drop(CALLBACKS.take());
    }
    FOUND.take()
}

/// Counts ``obj`` for the trace running on this thread until its region is
/// handed over or refused: the tracer's Python code calls it for whatever
/// it holds of the program's that the traced body may let go of (a copy of
/// an argument's items, taken before the body ran), so that the object dies
/// where the engine sees the code its death runs.
#[pyfunction]
pub fn _region_keep(obj: Bound<'_, PyAny>) {
    unsafe { keep(obj.as_ptr()) };
}

/// These function objects are the running trace's own weak-reference
/// callbacks (fx's table of tensors takes the entry of a tensor that died
/// out): a frame of one of them, entered where no instruction of the traced
/// body is in flight, is no code of the program's.
#[pyfunction]
pub fn _region_callbacks(funcs: Vec<Bound<'_, PyAny>>) {
    CALLBACKS.with_borrow_mut(|c| c.extend(funcs.into_iter().map(Bound::unbind)));
}
