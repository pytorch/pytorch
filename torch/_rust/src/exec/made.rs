// The containers, the iterators and the objects a traced run made.
//
// A region hands back a list, dict or set that was there before its trace
// as the object it is, and builds one that its run made anew on every call
// (torch/_rust_inductor.py's _held and _template). The reads cannot tell the
// two apart: a run reads what it made (its length, an item, all of it) as it
// reads anything else. The same of an iterator: one that was there before is
// the program's, with its position, and one the run made is made again over
// what it is over. And of an object a region can make again from its fields
// (a dataclass instance): the one a global holds is the program's, the one
// the run constructed is the run's.
//
// What it is given
//   - by the executor (run.rs): every single result of an instruction that
//     the traced body, or a frame under it, ran, once it is in its slot,
//     with the instruction and its operands; and every cell such an
//     instruction makes or stores into;
//   - by reads.rs: when a trace begins and ends.
// What it guarantees
//   - is(o) is true for a list, dict or set only if an instruction of the
//     run constructed it: a build instruction (a display, a comprehension),
//     a call of a class that gave an object of exactly that class, a call of
//     one of the callables that return a new container (super::makers), a
//     slice of an exact list, or an operator between exact builtin
//     containers that is not in place; and only if nothing but the slot
//     referred to the result at that moment;
//   - is(o) is true for an iterator only if the run made it: GET_ITER or
//     iter() of an object whose class iterates by the C code of a list, a
//     tuple, a dict, a dict view or a range (which gives a new iterator on
//     every call), with nothing but the slot referring to it; a zip or an
//     enumerate object that a call of its class gave; and the iterator
//     inside one of those over an argument of such a class;
//   - is(o) is true for any other object only if a call of its class under
//     the run gave it, and the class is one of Python whose objects `object`
//     allocates and `type` hands out: such a call gives a new object every
//     time (its __init__ ran under the run, whose records refer to the
//     object by then: no count is asked);
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
//     narrows it;
//   - that GET_ITER or iter() gives a new iterator: of an iterator it gives
//     the iterator itself, and a class's own __iter__ gives what it likes
//     (its frame is the run's, and what that makes is recorded there);
//   - that a call of a class gives a new object: a __new__ of the class or a
//     __call__ of its metaclass may hand out one that was there before (one
//     instance for all, a pool).

use std::cell::RefCell;
use std::collections::HashSet;
use std::sync::OnceLock;

use pyo3::ffi;
use pyo3::ffi::PyObject;

use super::ffi::tr_bound_function;
use crate::cfg::ir::Op;
use crate::cfg::opcodes;

#[derive(Default)]
struct Made {
    // Containers, iterators and objects of classes.
    objects: HashSet<usize>,
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

// Whether instruction `op`, handed `operands`, constructs its result `o`, a
// list, dict or set.
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

// Whether iterating `x` runs the C code of a list, a tuple, a dict, a dict
// view or a range, which makes a new iterator on every call (a class of
// Python that iterates by its own __iter__ does not).
unsafe fn iterable(x: *mut PyObject) -> bool {
    static SLOTS: OnceLock<HashSet<usize>> = OnceLock::new();
    let slots = SLOTS.get_or_init(|| {
        let classes = [&raw mut ffi::PyList_Type, &raw mut ffi::PyTuple_Type, &raw mut ffi::PyDict_Type, &raw mut ffi::PyDictKeys_Type, &raw mut ffi::PyDictValues_Type, &raw mut ffi::PyDictItems_Type, &raw mut ffi::PyRange_Type];
        classes.iter().filter_map(|c| unsafe { (**c).tp_iter }).map(|f| f as usize).collect()
    });
    !x.is_null() && (*ffi::Py_TYPE(x)).tp_iter.is_some_and(|f| slots.contains(&(f as usize)))
}

// The iterators instruction `op`, handed `operands`, made with its result
// `o`: `o` itself, and for a zip or an enumerate object those inside it that
// its class made over an argument (one that was handed in as an iterator is
// that iterator).
unsafe fn iterators(op: &Op, operands: &[*mut PyObject], o: *mut PyObject) -> Vec<*mut PyObject> {
    static ITER: OnceLock<usize> = OnceLock::new();
    let (callee, args, keywords) = match op {
        Op::GetIter => return if operands.first().is_some_and(|x| iterable(*x)) { vec![o] } else { vec![] },
        Op::Call { kw_names, .. } if operands.len() >= 2 => {
            let bound = operands[0].is_null();
            (operands[if bound { 1 } else { 0 }], &operands[if bound { 2 } else { 1 }..], kw_names.is_some())
        }
        _ => return vec![],
    };
    let class = ffi::Py_TYPE(o);
    let zip = class == &raw mut ffi::PyZip_Type;
    // Looked up without running Python code: a traced run is under way.
    let iter = *ITER.get_or_init(|| {
        let builtins = ffi::PyImport_AddModule(c"builtins".as_ptr());
        let f = if builtins.is_null() { builtins } else { ffi::PyObject_GetAttrString(builtins, c"iter".as_ptr()) };
        ffi::PyErr_Clear();
        ffi::Py_XDECREF(f);
        f as usize
    });
    if callee as usize == iter {
        return if args.len() == 1 && !keywords && iterable(args[0]) { vec![o] } else { vec![] };
    }
    if callee != class as *mut PyObject || !(zip || class == &raw mut ffi::PyEnum_Type) {
        return vec![];
    }
    // (zip, (iterators...)[, strict]) or (enumerate, (iterator, count)).
    let mut made = vec![o];
    let name = ffi::PyUnicode_InternFromString(c"__reduce__".as_ptr());
    let reduced = if name.is_null() { name } else { ffi::PyObject_CallMethodObjArgs(o, name, std::ptr::null_mut::<PyObject>()) };
    ffi::Py_XDECREF(name);
    if reduced.is_null() || ffi::PyTuple_Check(reduced) == 0 || ffi::PyTuple_GET_SIZE(reduced) < 2 || ffi::PyTuple_Check(ffi::PyTuple_GET_ITEM(reduced, 1)) == 0 {
        ffi::PyErr_Clear();
        ffi::Py_XDECREF(reduced);
        return made;
    }
    let inside = ffi::PyTuple_GET_ITEM(reduced, 1);
    // zip's only keyword is `strict`; enumerate's iterable comes first.
    let over = if zip { args.len() - keywords as usize } else { args.len().min(1) };
    for (k, x) in args.iter().take(over.min(ffi::PyTuple_GET_SIZE(inside) as usize)).enumerate() {
        if iterable(*x) {
            made.push(ffi::PyTuple_GET_ITEM(inside, k as ffi::Py_ssize_t));
        }
    }
    ffi::Py_DECREF(reduced);
    made
}

// Whether instruction `op`, handed `operands`, made `o` by calling its
// class: a class of Python that neither allocates its objects itself (a
// __new__ of its own, a C base with one) nor is called by a metaclass's own
// code, so that the object is one `object.__new__` just made.
unsafe fn instance(op: &Op, operands: &[*mut PyObject], o: *mut PyObject) -> bool {
    let callee = match op {
        Op::Call { .. } if operands.len() >= 2 => operands[if operands[0].is_null() { 1 } else { 0 }],
        Op::Bytecode { opcode: opcodes::CALL_FUNCTION_EX, .. } if !operands.is_empty() => operands[0],
        _ => return false,
    };
    let class = ffi::Py_TYPE(o);
    let slot = |f: Option<ffi::newfunc>| f.map_or(0, |f| f as usize);
    let call = |c: *mut ffi::PyTypeObject| (*c).tp_call.map_or(0, |f| f as usize);
    callee == class as *mut PyObject
        && ffi::PyType_HasFeature(class, ffi::Py_TPFLAGS_HEAPTYPE) != 0
        && slot((*class).tp_new) == slot((*(&raw mut ffi::PyBaseObject_Type)).tp_new)
        && call(ffi::Py_TYPE(callee)) == call(&raw mut ffi::PyType_Type)
}

// Instruction `op` of the traced run, handed `operands`, put `o` in its slot
// as its one result.
pub unsafe fn result(op: &Op, operands: &[*mut PyObject], o: *mut PyObject) {
    if o.is_null() {
        return;
    }
    let alone = ffi::Py_REFCNT(o) == 1;
    if container(o) {
        if alone && constructs(op, operands, o) {
            MADE.with_borrow_mut(|m| m.objects.insert(o as usize));
        }
    } else if ffi::PyIter_Check(o) != 0 {
        if alone {
            let made = iterators(op, operands, o);
            MADE.with_borrow_mut(|m| m.objects.extend(made.into_iter().map(|x| x as usize)));
        }
    } else if instance(op, operands, o) {
        MADE.with_borrow_mut(|m| m.objects.insert(o as usize));
    }
}

// An instruction of the traced run made the cell `c`, or stored into it.
pub fn cell(c: *mut PyObject) {
    MADE.with_borrow_mut(|m| m.cells.insert(c as usize));
}

// Whether the run of the trace on this thread made the container, the
// iterator or the object `o`.
pub fn is(o: *mut PyObject) -> bool {
    MADE.with_borrow(|m| m.objects.contains(&(o as usize)))
}

// Whether the run of the trace on this thread made or stored into cell `c`.
pub fn own_cell(c: *mut PyObject) -> bool {
    MADE.with_borrow(|m| m.cells.contains(&(c as usize)))
}
