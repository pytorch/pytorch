// The call of the autograd engine: a barrier, in a region and out of one.
//
// A call into C code is where a region ends unless the engine knows what the
// call does. `run_backward` of the autograd engine (what Tensor.backward and
// torch.autograd.grad end in) is such a call that also calls back into
// Python. Two things follow.
//
// It is no part of a region. A region is the torch ops of one traced run,
// made again; a backward runs torch ops too, and what it does besides is
// none of them: it writes `.grad`, frees the buffers of the graph it went
// through, runs the hooks on its way. A region made of its ops would do none
// of that, and a trace given up after the engine started has left behind
// what its block, run again, then does a second time. A region gets its
// backward from the joint graph a pass list makes of its forward, not from
// the engine's callbacks.
//
// What it calls back into does not run on the executor. That is a hook, the
// backward of an autograd.Function, and the graph a lowering made for a
// region's backward, which AOT Autograd compiles when the backward first
// runs: inside the program's call. The code of a compiler at work is not the
// program's: observed and made regions of, it fails or changes what it
// compiles. So the executor runs the engine's call outside the executor, as
// PyTorch runs it. Given up: there is no region inside a hook, inside the
// backward of an autograd.Function, or inside a graph made for a backward.
//
// What it is given
//   - a callable: the callee the optimizer recorded for a call (mod.rs's
//     pure_instr), or the one the executor is about to call (run.rs).
// What it guarantees
//   - is(f): whether a call of f starts the engine, bound to it or not;
//   - refuse(): the trace running on this thread is refused, as any other is
//     (region.rs's refuse), in place of the call: the engine has not run.
// What it may not assume
//   - that the optimizer saw the call: it judges the callees and the blocks
//     that ran while it observed, and the traced run may be the first to come
//     by here. So the executor asks before every call it makes (run.rs's
//     slow_step): in a traced body the call is refused, anywhere else it
//     runs outside the executor.

use std::sync::OnceLock;

use pyo3::ffi;
use pyo3::ffi::PyObject;
use pyo3::prelude::*;

use super::ffi::tr_bound_function;

// The engine's class and its run_backward as the class has it, by address
// (0: none).
fn engine() -> (usize, usize) {
    static ENGINE: OnceLock<(usize, usize)> = OnceLock::new();
    *ENGINE.get_or_init(|| {
        let py = unsafe { Python::assume_attached() };
        let class = py.import("torch._C").and_then(|m| m.getattr("_ImperativeEngine"));
        let run = class.as_ref().ok().and_then(|c| c.getattr("__dict__").and_then(|d| d.get_item("run_backward")).ok());
        match (class, run) {
            (Ok(c), Some(r)) => (c.as_ptr() as usize, r.as_ptr() as usize),
            _ => (0, 0),
        }
    })
}

// Looks the engine up before a body is traced: not a read of the body's.
pub fn ready() {
    engine();
}

// Whether a call of `callable` starts the autograd engine.
pub unsafe fn is(callable: *mut PyObject) -> bool {
    let (class, run) = engine();
    if callable.is_null() || run == 0 {
        return false;
    }
    if callable as usize == run {
        return true;
    }
    if ffi::PyCFunction_Check(callable) == 0 {
        return false;
    }
    let owner = ffi::PyCFunction_GET_SELF(callable);
    !owner.is_null() && ffi::PyObject_TypeCheck(owner, class as *mut ffi::PyTypeObject) != 0 && tr_bound_function(callable) as usize == run
}

// Refuses the trace running on this thread where its body calls the engine.
pub fn refuse() {
    super::region::refuse(unsafe { Python::assume_attached() }, "it runs a backward, which writes .grad and frees the graph it went through", false);
}
