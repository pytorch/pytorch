//! The state around a call that is no argument of it: what a region is
//! traced under and entered under.
//!
//! Two parts. The WORD (region_thread_state of csrc/tensor_ffi.h) is state a
//! region is for: autocast, the default dtype, the determinism and backend
//! flags, grad mode, inference mode, the current CUDA device. A region is
//! entered only under the word of its trace (compared with the reads, see
//! reads.rs's `Reads::check`). Everything else must be NEUTRAL, at the
//! start of a trace and at every entry: no dispatch mode and no torch
//! function mode, no functorch transform, no forward-AD level, no
//! saved-tensor hooks, JIT and fx tracing off, anomaly mode off, autocast
//! off for the device types the word does not name. Under any of them the
//! program's own calls are to be seen, wrapped or recorded one by one, which
//! a region's replay is not: no trace starts and no region is entered, and
//! the code runs as written.
//!
//! Given: the thread's state, read under the GIL. No Python code runs.
//!
//! Guarantees: `not_neutral()` is None only if every item above is neutral,
//! or is the engine's own: while the body of a trace runs (`Own`), the
//! dispatch modes, the function modes and the fx flag the tracer itself put
//! on are what "none" means, so a region inside a traced body is still
//! entered, and anything the program adds on top is not neutral.
//!
//! May not assume: that state named in neither part cannot change a result
//! (the number of threads does, bit for bit: a kernel reads it when it
//! runs, under the region as under eager).

use std::cell::Cell;

use pyo3::prelude::*;

use crate::bindings::ffi as tffi;

thread_local! {
    // What the tracer has on while a traced body runs on this thread.
    static OWN: Cell<tffi::AmbientModes> = const { Cell::new(tffi::AmbientModes { dispatch_modes: 0, function_modes: 0, fx_tracing: false }) };
}

// While it lives, the modes on now are the engine's own: taken where the
// traced body starts to run, under the tracer.
pub struct Own(tffi::AmbientModes);

impl Own {
    pub fn start() -> Own {
        let mut now = tffi::AmbientModes::default();
        tffi::region_modes(&mut now);
        Own(OWN.replace(now))
    }
}

impl Drop for Own {
    fn drop(&mut self) {
        OWN.set(self.0);
    }
}

// What is not neutral, if anything is.
pub fn not_neutral() -> Option<&'static str> {
    Some(match tffi::region_not_neutral(&OWN.get()) {
        0 => return None,
        1 => "a dispatch mode is on",
        2 => "a torch function mode is on",
        3 => "a functorch transform is on",
        4 => "a forward-AD level is open",
        5 => "saved-tensor hooks are on",
        6 => "the JIT is tracing",
        7 => "torch.fx is tracing",
        8 => "anomaly mode is on",
        // 9, and whatever the C++ side comes to name: not neutral.
        _ => "autocast is on for another device type",
    })
}

/// What of the thread's state is not neutral (see the module), or None.
#[pyfunction]
pub fn _not_neutral() -> Option<&'static str> {
    not_neutral()
}
