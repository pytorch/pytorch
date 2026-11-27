// Which Python frame an instruction of a recorded run may enter (record.rs
// asks; nothing else does).
//
// What it is given: an instruction that is no entry, before it runs (its row
// of the rule table if it has one, what it is, where it runs, its callee);
// then each frame entered while it is in flight in the innermost activation
// of the run.
// What it guarantees
//   - a frame is the instruction's only POSITIVELY: its function object is,
//     by identity, a function the instruction names, and no frame of that
//     function was entered under the instruction before. Functions are named
//     for five forms and for nothing else:
//       the call of an exact Python function: that function;
//       the call of an exact bound method of an exact Python function: it;
//       the call of an object whose class, one written in Python, has an
//       exact Python function under __call__: it;
//       an attribute or method load: the exact Python function that the
//       class of what the object's class has under the name, one written in
//       Python, has under __get__ (a descriptor written in Python), and
//       the exact Python function the object's class, one written in
//       Python, has under __getattr__ (the fallback of the lookup);
//   - every other frame is refused, with why: under an instruction whose row
//     enters none; under one that names no function (a class that is called,
//     a C callee with no row, a property, a store, an iterator); a frame of
//     another function, or a second frame of a named one (a finalizer that
//     ran when the callee's frame ended);
//   - nothing of the program runs: a class is asked by the type's own lookup
//     (csrc/exec_ops.c), a callable by its exact type;
//   - a named function is HELD while its instruction is in flight: a frame's
//     function is compared with a function that lives, never with an address
//     another function could have been made at. (A callee that is an operand
//     is held by the executor's value for as long anyway; what a class has
//     under a name is held by the class only, and the program can take it
//     away under the instruction.) record.rs lets go where the instruction
//     ends, outside its borrow of the recordings: a function that dies there
//     may run a finalizer.
// What it may not assume
//   - that a row which names frames (Enters::Named) says which: none does
//     yet, so such an instruction names no function here.

use std::sync::OnceLock;

use pyo3::ffi::{self, PyObject};
use pyo3::prelude::*;

use crate::cfg::ir::Op;
use crate::exec::ffi::{tr_bound_function, tr_frame_func, tr_type_function, tr_type_heap, tr_type_lookup, Frame, PyMethod_Type};
use crate::exec::rules::{Enters, Row, Site};

#[derive(Default)]
pub struct Frames {
    // The functions the instruction names, held, and whether a frame of
    // each was entered under it.
    named: [Option<Py<PyAny>>; 2],
    entered: [bool; 2],
    // Why no frame is the instruction's, if it named none.
    none: Option<&'static str>,
}

const UNNAMED: &str = "which names no function that the frame could be compared with";

unsafe fn function(f: *mut PyObject) -> Option<Py<PyAny>> {
    unsafe { (!f.is_null() && ffi::Py_TYPE(f) == &raw mut ffi::PyFunction_Type).then(|| Py::from_borrowed_ptr(Python::assume_attached(), f)) }
}

// What the class of `descriptor`, if it is one written in Python, has under
// __get__, if that is a Python function.
unsafe fn getter(descriptor: *mut PyObject) -> Option<Py<PyAny>> {
    static GET: OnceLock<usize> = OnceLock::new();
    unsafe {
        let name = *GET.get_or_init(|| ffi::PyUnicode_InternFromString(c"__get__".as_ptr()) as usize) as *mut PyObject;
        let class = ffi::Py_TYPE(descriptor);
        if descriptor.is_null() || name.is_null() || tr_type_heap(class) == 0 { None } else { function(tr_type_lookup(class, name)) }
    }
}

impl Frames {
    pub unsafe fn of(row: Option<&Row>, op: &Op, site: &Site, callee: Option<*mut PyObject>) -> Frames {
        unsafe {
            if let Some(row) = row {
                return Frames { none: Some(if row.enters == Enters::Named { UNNAMED } else { "which enters none" }), ..Frames::default() };
            }
            let named = match (callee, op) {
                (Some(f), _) if ffi::Py_TYPE(f) == &raw mut PyMethod_Type => [function(tr_bound_function(f)), None],
                (Some(f), _) if ffi::Py_TYPE(f) == &raw mut ffi::PyFunction_Type => [function(f), None],
                (Some(f), _) => [function(tr_type_function(f, 0)), None],
                (None, Op::LoadAttr(n) | Op::LoadMethod(n)) if site.operands.first().is_some_and(|o| !o.is_null()) => {
                    let object = site.operands[0];
                    [getter(tr_type_lookup(ffi::Py_TYPE(object), *site.names.add(*n as usize))), function(tr_type_function(object, 1))]
                }
                _ => [None, None],
            };
            let none = named.iter().all(Option::is_none).then_some(UNNAMED);
            Frames { named, none, ..Frames::default() }
        }
    }

    // Ok: `frame` is one the instruction called. Err: why it is not. (No
    // reference is let go of here: the caller holds the recordings.)
    pub unsafe fn enter(&mut self, frame: *mut Frame) -> Result<(), &'static str> {
        if let Some(why) = self.none {
            return Err(why);
        }
        let function = unsafe { tr_frame_func(frame) };
        let is = |at: &usize| !self.entered[*at] && self.named[*at].as_ref().is_some_and(|named| named.as_ptr() == function);
        match [0, 1].iter().find(|at| is(at)) {
            Some(at) => {
                self.entered[*at] = true;
                Ok(())
            }
            None if (0..2).all(|at| self.named[at].is_none() || self.entered[at]) => Err("whose callee's frame had been entered already"),
            None => Err("whose callee is another function"),
        }
    }
}
