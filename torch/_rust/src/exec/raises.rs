// An exception that a frame under a traced body raises, and who catches it.
//
// A frame that returns hands the instruction that entered it a value. A
// frame that ends in `raise X(...)` hands the C code that entered it an
// exception, and where that C code catches it the instruction goes on with
// an answer, as after a return: `hasattr(m, name)` and `getattr(m, name,
// default)` for the AttributeError of a `__getattr__`, a loop or `in` for
// the IndexError of a `__getitem__`. Such a frame only computed if its
// blocks did, and what it read (the keys it did not find) is checked by the
// region as any frame's reads. An exception that leaves the instruction is
// another matter: the program's own handler takes it, or nobody does, and
// that is as it was before this rule.
//
// What it is given
//   - by the optimizer (mod.rs's pure_instr): a call in the code of a
//     callee and its callee (made);
//   - by covered.rs's step(), before every instruction of a traced body or
//     of a frame under it: whether it calls an exception class (asked);
//   - by the frame hook (mod.rs's eval_frame): a frame under a traced body
//     that ended with an exception set (left);
//   - by covered.rs: a trace starts (begin), is over, or a frame runs with
//     no trace on (over).
// What it guarantees
//   - made() is true only for a call of an exception class that C code
//     makes and initializes (its metaclass is type, no class of the
//     program's in its MRO has a __new__ or an __init__), that is the last
//     instruction of its block, and whose result that block raises, alone,
//     out of its frame: the call reads its arguments and no more, and no
//     instruction of the frame runs between it and the frame's end;
//   - asked() is Some(why) when an exception class was called under the
//     trace on this thread, the frame that called it ended with an
//     exception before another instruction was asked about, and something
//     still refers to that exception now: the program has it (a handler
//     took it, a cleanup runs for it). One that C code caught and dropped
//     is gone by then;
//   - the exception is referred to from here no longer than until the next
//     instruction asked about, the next trace, or the next frame that runs
//     with no trace on.
// What it may not assume
//   - that C code which catches an exception does nothing else with it
//     (`sys.unraisablehook` reports the one a finalizer raises);
//   - that an exception of any other origin (a bare `raise X`, one C code
//     raises, one made earlier and raised later) is seen here: those are
//     judged and traced as before.

use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicU32, Ordering};

use pyo3::ffi;
use pyo3::ffi::PyObject;

use crate::cfg::ir::{TermUnwind, Terminator};

use super::{Compiled, Prepared, ValueId};

thread_local! {
    // The instruction asked about last on this thread calls an exception
    // class.
    static MAKING: Cell<bool> = const { Cell::new(false) };
    // The exception a frame that made one ended with (owned), and the
    // frame's function.
    static LEFT: Cell<*mut PyObject> = const { Cell::new(std::ptr::null_mut()) };
    static BY: RefCell<String> = const { RefCell::new(String::new()) };
}

// How many threads hold an exception in LEFT (the frame hook asks at every
// frame that runs with no trace on).
static HELD: AtomicU32 = AtomicU32::new(0);

// Whether `f` is a class of exceptions whose objects C code makes and
// initializes from the call's arguments.
pub unsafe fn class(f: *mut PyObject) -> bool {
    if ffi::Py_TYPE(f) != std::ptr::addr_of_mut!(ffi::PyType_Type) {
        return false;
    }
    let tp = f as *mut ffi::PyTypeObject;
    let mro = (*tp).tp_mro;
    if ffi::PyType_IsSubtype(tp, ffi::PyExc_BaseException as *mut ffi::PyTypeObject) == 0 || mro.is_null() {
        return false;
    }
    (0..ffi::PyTuple_GET_SIZE(mro)).all(|i| {
        let k = ffi::PyTuple_GET_ITEM(mro, i) as *mut ffi::PyTypeObject;
        if ffi::PyType_HasFeature(k, ffi::Py_TPFLAGS_HEAPTYPE) == 0 {
            return true;
        }
        let own = (*k).tp_dict;
        let lacks = |name: &std::ffi::CStr| ffi::PyDict_GetItemString(own, name.as_ptr()).is_null();
        !own.is_null() && lacks(c"__new__") && lacks(c"__init__")
    })
}

// Whether instr v of `prep`, a call of `f`, makes an exception that its
// block raises out of the frame at once (see the guarantees above).
pub fn made(prep: &Prepared, v: ValueId, f: *mut PyObject) -> bool {
    let cfg = &prep.cfg;
    let raised_class = unsafe { class(f) };
    raised_class
        && cfg.block_ids().any(|b| {
            let Some(ver) = cfg[b].versions.first() else { return false };
            let raised = match &cfg[b].term {
                Terminator::Raise { args, unwind: TermUnwind::Propagate } if args.len() == 1 => ver.outputs.get(args[0] as usize),
                _ => None,
            };
            ver.body.last() == Some(&v) && raised == Some(&v)
        })
}

fn drop_left() -> bool {
    let exc = LEFT.replace(std::ptr::null_mut());
    if exc.is_null() {
        return false;
    }
    HELD.fetch_sub(1, Ordering::Relaxed);
    // What still refers to it besides this reference.
    let kept = unsafe { ffi::Py_REFCNT(exc) > 1 };
    unsafe { ffi::Py_DECREF(exc) };
    kept
}

// Before an instruction of a traced body or of a frame under it runs;
// `makes`: it calls an exception class. Some(why) if the program has the
// exception that the frame of the last such call ended with.
pub fn asked(makes: bool) -> Option<String> {
    MAKING.set(makes);
    drop_left().then(|| {
        let by = BY.with_borrow(|b| b.clone());
        format!("an exception that {by} raises is taken by the code's own handler: a region goes through a raise only where C code catches it (hasattr, getattr with a default, a loop's end)")
    })
}

// A frame of `c` under a traced body ended with an exception set.
pub unsafe fn left(c: &Compiled) {
    if !MAKING.replace(false) && LEFT.get().is_null() {
        return;
    }
    // Through the frames above the one that made it, the same exception.
    let exc = ffi::PyErr_GetRaisedException();
    if exc.is_null() {
        return;
    }
    let old = LEFT.replace(exc);
    ffi::Py_INCREF(exc);
    ffi::PyErr_SetRaisedException(exc);
    if old.is_null() {
        HELD.fetch_add(1, Ordering::Relaxed);
        BY.with_borrow_mut(|b| b.clone_from(&c.qualname));
    } else {
        ffi::Py_DECREF(old);
    }
}

// A trace starts: nothing of the one before is left.
pub fn begin() {
    MAKING.set(false);
    over();
}

// A trace is over, or a frame runs with no trace on: nothing is held any
// longer.
#[inline]
pub fn over() {
    if HELD.load(Ordering::Relaxed) != 0 {
        drop_left();
    }
}
