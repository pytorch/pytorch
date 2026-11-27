//! The state around a call that is no argument of it: what a region is
//! traced under and entered under.
//!
//! Two parts. The WORD (region_word of csrc/tensor_ffi.h) is state a
//! region is for: autocast, the default dtype, the determinism and backend
//! flags, the float32 precision of every backend and operator, the flags
//! of autograd's thread state (grad mode, inference mode and the four
//! others), the current CUDA device. A region is
//! entered only under the word of its trace: every variant holds one, and
//! region.rs's `Variant::enter`, the one place a region runs from, compares
//! it.
//!
//! Everything else must be NEUTRAL, at the start of a trace, at its end
//! and at every entry. Neutral is a positive statement (region_not_neutral
//! of csrc/tensor_ffi.cpp): the thread's state, as torch itself records it
//! (at::ThreadLocalState, member by member), is what a thread starts with,
//! or what the word says it is. The dispatcher's local key sets are
//! compared whole against the default ones as the word moves them, so a
//! key nobody listed, one added next year too, is not neutral; so are a
//! dispatch mode, a torch function mode or disabled overrides, a functorch
//! transform, a fake tensor mode, the Python dispatcher, a profiler or any
//! other callback on operators, saved-tensor hooks, node-creation hooks;
//! and, kept elsewhere, a forward-AD level, JIT and fx tracing, anomaly
//! mode.
//! Under any of them the program's own calls are to be seen, wrapped or
//! recorded one by one, which a region's replay is not: no trace starts and
//! no region is entered, and the code runs as written.
//!
//! Given: the thread's state, read under the GIL. No Python code runs.
//!
//! Guarantees: `state()` gives a word only if all of the above holds. It
//! makes no allowance for the engine's own tracer: no region is entered
//! and none is traced on a thread while a trace runs on it (region.rs's
//! `tracing()`), and the tracer asks before it starts and after it ended.
//!
//! May not assume: that state named in neither part cannot change a result
//! (the number of threads does, bit for bit: a kernel reads it when it
//! runs, under the region as under eager). What cannot be asked and is not
//! looked at: the Python objects torch keeps per thread under names
//! (at::impl::ThreadLocalPythonObjects), the current CUDA stream.

use pyo3::prelude::*;

use crate::bindings::ffi as tffi;

// The word of the thread now. None: some of it cannot be read, or has a
// value the word has no room for; under that no trace starts and no region
// is entered.
pub fn word() -> Option<tffi::Word> {
    let mut w = tffi::Word::default();
    tffi::region_word(&mut w).then_some(w)
}

// The state of the thread, asked once: its word if it is neutral, else
// what is not neutral.
pub fn state() -> Result<tffi::Word, &'static str> {
    let Some(word) = word() else { return Err("the state of the thread cannot be read") };
    Err(match tffi::region_not_neutral() {
        0 => return Ok(word),
        1 => "a dispatch mode is on",
        2 => "a torch function mode is on",
        3 => "a functorch transform is on",
        4 => "a forward-AD level is open",
        5 => "saved-tensor hooks are on or disabled",
        6 => "the JIT is tracing",
        7 => "torch.fx is tracing",
        8 => "anomaly mode is on",
        9 => "autocast is on for another device type",
        10 => "a profiler or another callback on operators is on",
        11 => "the dispatcher's keys of the thread are not the default ones",
        12 => "torch function overrides are disabled",
        13 => "a fake tensor mode is on",
        14 => "the Python dispatcher is on",
        15 => "node-creation hooks or a graph-execution group are on",
        16 => "a flag of functionalization or DTensor is not at its default",
        // 255, and whatever the C++ side comes to name: not neutral.
        _ => "the state of the thread cannot be asked",
    })
}

// What is not neutral, if anything is.
pub fn not_neutral() -> Option<&'static str> {
    state().err()
}

/// The ambient word of the thread, as two ints (see the module), or None if
/// it cannot be read: equal words are one state to every region.
#[pyfunction]
pub fn _ambient_word() -> Option<(u64, u64)> {
    word().map(|w| (w.a, w.b))
}

/// What of the thread's state is not neutral (see the module), or None.
#[pyfunction]
pub fn _not_neutral() -> Option<&'static str> {
    not_neutral()
}
