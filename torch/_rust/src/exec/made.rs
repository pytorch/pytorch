// The containers a traced run made.
//
// A region hands back a list, dict or set that was there before its trace
// as the object it is, and builds one that its run made anew on every call
// (torch/_rust_inductor.py's _held and _template). The reads cannot tell the
// two apart: a run reads what it made (its length, an item, all of it) as it
// reads anything else.
//
// What it is given
//   - by the executor (run.rs): every single result of an instruction that
//     the traced body, or a frame under it, ran, once it is in its slot,
//     with the instruction and its first two operands; and every cell such
//     an instruction makes or stores into;
//   - by reads.rs: when a trace begins and ends.
// What it guarantees
//   - is(o) is true only for a list, dict or set that an instruction of the
//     run constructed: a build instruction (a display, a comprehension), a
//     call of a class that gave an object of exactly that class, a call of
//     one of the callables that return a new container (super::makers), a
//     slice of an exact list, or an operator between exact builtin
//     containers that is not in place; and only if nothing but the slot
//     referred to the result at that moment;
//   - own_cell(c) is true only for a cell the run made or stored into: what
//     it holds is the run's, not something that was there before.
// What it may not assume
//   - that a container for which is(o) is false was there before: one that
//     came about any other way (C code that keeps what it makes in something
//     else, as a defaultdict does for a missing key; an item an iterator
//     gave; a starred target) is neither made nor known, and a region that
//     hands it on is refused;
//   - that an instruction's result is new because nothing else refers to
//     it: a call that takes an item OUT of a container that was there before
//     (pop, popleft, next) gives an old object with one reference. The rule
//     is by what the instruction or the callee is, and the count only
//     narrows it.

use std::cell::RefCell;
use std::collections::HashSet;

use pyo3::ffi;
use pyo3::ffi::PyObject;

use super::ffi::tr_bound_function;
use crate::cfg::ir::Op;
use crate::cfg::opcodes;

#[derive(Default)]
struct Made {
    containers: HashSet<usize>,
    cells: HashSet<usize>,
}

thread_local! {
    static MADE: RefCell<Made> = RefCell::new(Made::default());
}

// A trace begins, or is over, on this thread.
pub fn clear() {
    MADE.with_borrow_mut(|m| *m = Made::default());
}

unsafe fn container(o: *mut PyObject) -> bool {
    ffi::PyList_Check(o) != 0 || ffi::PyDict_Check(o) != 0 || ffi::PyAnySet_Check(o) != 0
}

// An object of a builtin class itself, whose operators run C code that
// gives a new object.
unsafe fn exact(o: *mut PyObject) -> bool {
    !o.is_null()
        && (ffi::PyList_CheckExact(o) != 0
            || ffi::PyDict_CheckExact(o) != 0
            || ffi::PyAnySet_CheckExact(o) != 0
            || ffi::PyTuple_CheckExact(o) != 0
            || ffi::PyLong_CheckExact(o) != 0)
}

// Whether instruction `op`, handed `operands` (its first two), constructs
// its result `o`.
unsafe fn constructs(op: &Op, operands: &[*mut PyObject], o: *mut PyObject) -> bool {
    let at = |k: usize| operands.get(k).copied().unwrap_or(std::ptr::null_mut());
    match op {
        Op::Bytecode { opcode, .. } => match *opcode {
            opcodes::BUILD_LIST | opcodes::BUILD_MAP | opcodes::BUILD_SET | opcodes::BUILD_CONST_KEY_MAP => true,
            opcodes::BINARY_SLICE => !at(0).is_null() && ffi::PyList_CheckExact(at(0)) != 0,
            opcodes::BINARY_SUBSCR => !at(0).is_null() && !at(1).is_null() && ffi::PyList_CheckExact(at(0)) != 0 && ffi::PySlice_Check(at(1)) != 0,
            _ => false,
        },
        // In place (13 on), the result is the left operand.
        Op::Binary(k) => *k < 13 && exact(at(0)) && exact(at(1)),
        Op::Call { .. } => {
            let callee = if at(0).is_null() { at(1) } else { at(0) };
            !callee.is_null() && (callee == ffi::Py_TYPE(o) as *mut PyObject || super::makers().contains(&(tr_bound_function(callee) as usize)))
        }
        _ => false,
    }
}

// Instruction `op` of the traced run, handed `operands` (its first two), put
// `o` in its slot as its one result.
pub unsafe fn result(op: &Op, operands: &[*mut PyObject], o: *mut PyObject) {
    if !o.is_null() && ffi::Py_REFCNT(o) == 1 && container(o) && constructs(op, operands, o) {
        MADE.with_borrow_mut(|m| m.containers.insert(o as usize));
    }
}

// An instruction of the traced run made the cell `c`, or stored into it.
pub fn cell(c: *mut PyObject) {
    MADE.with_borrow_mut(|m| m.cells.insert(c as usize));
}

// Whether the run of the trace on this thread made the container `o`.
pub fn is(o: *mut PyObject) -> bool {
    MADE.with_borrow(|m| m.containers.contains(&(o as usize)))
}

// Whether the run of the trace on this thread made or stored into cell `c`.
pub fn own_cell(c: *mut PyObject) -> bool {
    MADE.with_borrow(|m| m.cells.contains(&(c as usize)))
}
