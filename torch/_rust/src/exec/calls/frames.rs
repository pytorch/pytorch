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
//     (csrc/exec_ops.c), a callable by its exact type.
// What it may not assume
//   - that a row which names frames (Enters::Named) says which: none does
//     yet, so such an instruction names no function here;
//   - that a named function lives while the instruction is in flight: it is
//     known by its address, which the frame's function is compared with.

use std::sync::OnceLock;

use pyo3::ffi::{self, PyObject};

use crate::cfg::ir::Op;
use crate::exec::ffi::{tr_bound_function, tr_frame_func, tr_type_function, tr_type_heap, tr_type_lookup, Frame, PyMethod_Type};
use crate::exec::rules::{Enters, Row, Site};

pub struct Frames {
    // The functions whose frame the instruction may still enter (0: none).
    named: [usize; 2],
    // Why no frame is the instruction's, if it named none.
    none: Option<&'static str>,
}

const UNNAMED: &str = "which names no function that the frame could be compared with";

unsafe fn function(f: *mut PyObject) -> usize {
    if !f.is_null() && unsafe { ffi::Py_TYPE(f) } == &raw mut ffi::PyFunction_Type { f as usize } else { 0 }
}

// What the class of `descriptor`, if it is one written in Python, has under
// __get__, if that is a Python function.
unsafe fn getter(descriptor: *mut PyObject) -> usize {
    static GET: OnceLock<usize> = OnceLock::new();
    unsafe {
        let name = *GET.get_or_init(|| ffi::PyUnicode_InternFromString(c"__get__".as_ptr()) as usize) as *mut PyObject;
        let class = ffi::Py_TYPE(descriptor);
        if name.is_null() || tr_type_heap(class) == 0 { 0 } else { function(tr_type_lookup(class, name)) }
    }
}

impl Frames {
    pub unsafe fn of(row: Option<&Row>, op: &Op, site: &Site, callee: Option<*mut PyObject>) -> Frames {
        unsafe {
            if let Some(row) = row {
                return Frames { named: [0, 0], none: Some(if row.enters == Enters::Named { UNNAMED } else { "which enters none" }) };
            }
            let named = match (callee, op) {
                (Some(f), _) if ffi::Py_TYPE(f) == &raw mut ffi::PyFunction_Type => [f as usize, 0],
                (Some(f), _) if ffi::Py_TYPE(f) == &raw mut PyMethod_Type => [function(tr_bound_function(f)), 0],
                (Some(f), _) => [function(tr_type_function(f, 0)), 0],
                (None, Op::LoadAttr(n) | Op::LoadMethod(n)) if site.operands.first().is_some_and(|o| !o.is_null()) => {
                    let object = site.operands[0];
                    let found = tr_type_lookup(ffi::Py_TYPE(object), *site.names.add(*n as usize));
                    [if found.is_null() { 0 } else { getter(found) }, function(tr_type_function(object, 1))]
                }
                _ => [0, 0],
            };
            Frames { named, none: (named == [0, 0]).then_some(UNNAMED) }
        }
    }

    // Ok: `frame` is one the instruction called. Err: why it is not.
    pub unsafe fn enter(&mut self, frame: *mut Frame) -> Result<(), &'static str> {
        if let Some(why) = self.none {
            return Err(why);
        }
        let function = unsafe { tr_frame_func(frame) } as usize;
        match self.named.iter().position(|named| *named != 0 && *named == function) {
            Some(at) => {
                self.named[at] = 0;
                Ok(())
            }
            None if self.named == [0, 0] => Err("whose callee's frame had been entered already"),
            None => Err("whose callee is another function"),
        }
    }
}
