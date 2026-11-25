// Stand-ins for a region's scalar inputs, and the places where their value
// is read without them seeing it.
//
// While torch.vm traces a region, each float argument is an instance of a
// float subclass (torch._rust_inductor._Scalar): its own methods record
// arithmetic as graph nodes, and note a comparison, a hash or a conversion
// by checking the region on the argument's value. C code that is handed the
// object reads the C double inside it without calling any of those methods
// (`math.sqrt(t)`, `float(t)`, `"%f" % t`, `1j * t`), and what it returns is
// a plain value that the trace would take for a constant.
//
// Given: every instruction the executor runs while a body is traced, with
// its operands, before it runs (`before`); and for a call that runs C code
// for the callee itself, its arguments and its result (`handed`, `called`).
//
// Guarantees: a stand-in that is an operand of an instruction (or sits in an
// exact tuple, list or dict that is one, a few levels deep) and that the
// instruction hands to C code is reported to torch._rust_inductor, unless
// that code is known to leave its value alone or to read it through the
// stand-in's own methods:
//   - an operator whose other operand is an int, a float, a bool, another
//     stand-in (these defer to the stand-in's method) or a tensor (the
//     trace's torch-function mode sees the operation);
//   - a store or a load that only moves the object (a local, an item of a
//     list, a dict or a tuple, an attribute held in a dict or a slot);
//   - a call that enters a Python frame for the callee itself;
//   - a call of C code that torch._rust_inductor knows (`_c_call` decides:
//     torch's own functions, builtins that move the object or call its
//     methods, and the functions of `math` that compute a float from
//     floats, whose result it makes a stand-in computed in the graph).
// For what is reported the region is checked on the value of the inputs the
// stand-in came from: a later call with another value gets a region of its
// own.
//
// May not assume: that it sees a stand-in C code comes by some other way
// (out of a generator the code is handed, through an attribute of an object
// that is an operand, deeper in a container than DEPTH).

use pyo3::ffi;
use pyo3::ffi::PyObject;
use pyo3::prelude::*;

use super::ffi::*;
use crate::bindings::ffi as tffi;
use crate::cfg::ir::{Instr, Op};
use crate::cfg::opcodes;

// How deep in the containers that are operands a stand-in is looked for.
const DEPTH: u8 = 4;

// The stand-in's class, by address (0: torch._rust_inductor isn't there).
fn class() -> usize {
    thread_local! {
        static CLASS: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) };
    }
    let known = CLASS.get();
    if known != usize::MAX {
        return known;
    }
    let py = unsafe { Python::assume_attached() };
    let found = py.import("torch._rust_inductor").and_then(|m| m.getattr("_Scalar")).map_or(0, |c| c.as_ptr() as usize);
    CLASS.set(found);
    found
}

// Whether `o` (NULL allowed) is a stand-in for a scalar input.
pub unsafe fn stand_in(o: *mut PyObject) -> bool {
    !o.is_null() && ffi::PyFloat_Check(o) != 0 && ffi::PyFloat_CheckExact(o) == 0 && ffi::Py_TYPE(o) as usize == class()
}

// The stand-ins `o` is or holds, appended to `out`.
unsafe fn find(o: *mut PyObject, depth: u8, out: &mut Vec<*mut PyObject>) {
    if o.is_null() {
        return;
    }
    if stand_in(o) {
        out.push(o);
    } else if depth < DEPTH && ffi::PyTuple_CheckExact(o) != 0 {
        for i in 0..ffi::PyTuple_GET_SIZE(o) {
            find(ffi::PyTuple_GET_ITEM(o, i), depth + 1, out);
        }
    } else if depth < DEPTH && ffi::PyList_CheckExact(o) != 0 {
        for i in 0..ffi::PyList_GET_SIZE(o) {
            find(ffi::PyList_GET_ITEM(o, i), depth + 1, out);
        }
    } else if depth < DEPTH && ffi::PyDict_CheckExact(o) != 0 {
        let (mut pos, mut k, mut v) = (0, std::ptr::null_mut(), std::ptr::null_mut());
        while ffi::PyDict_Next(o, &mut pos, &mut k, &mut v) != 0 {
            find(k, depth + 1, out);
            find(v, depth + 1, out);
        }
    }
}

// Calls torch._rust_inductor.`name` with these objects; what it raises is
// written out and dropped (the trace goes on as it would have).
unsafe fn tell<'py>(py: Python<'py>, name: &str, args: Vec<Bound<'py, PyAny>>) -> Option<Bound<'py, PyAny>> {
    let r = super::outside_executor(|| py.import("torch._rust_inductor").and_then(|m| m.getattr(name)).and_then(|f| f.call1(pyo3::types::PyTuple::new(py, args)?)));
    r.map_err(|e| e.write_unraisable(py, None)).ok()
}

// These stand-ins are read by value.
unsafe fn read(found: &[*mut PyObject]) {
    let py = Python::assume_attached();
    let objs = found.iter().map(|o| Bound::from_borrowed_ptr(py, *o)).collect();
    tell(py, "_scalars_read", objs);
}

// Whether an operator's other operand leaves the reading to the stand-in's
// own method, or is a tensor.
unsafe fn defers(other: *mut PyObject) -> bool {
    other.is_null() || ffi::PyFloat_CheckExact(other) != 0 || ffi::PyLong_CheckExact(other) != 0 || ffi::PyBool_Check(other) != 0 || stand_in(other) || !tffi::tensor_by_type(other as usize).is_null()
}

// Whether `container` is one of the builtin containers, which take an item
// or a key as an object (hashing and comparing it by its own methods), or a
// tensor.
unsafe fn holds_objects(container: *mut PyObject) -> bool {
    !container.is_null()
        && (ffi::PyList_CheckExact(container) != 0
            || ffi::PyTuple_CheckExact(container) != 0
            || ffi::PyDict_CheckExact(container) != 0
            || ffi::PyAnySet_CheckExact(container) != 0
            || !tffi::tensor_by_type(container as usize).is_null())
}

// The instruction `ins`, with these operands (in the order of its args; one
// of them is a stand-in), is about to run while a body is traced. `name`:
// the attribute a StoreAttr stores.
pub unsafe fn before(ins: &Instr, operands: &[*mut PyObject], name: *mut PyObject) {
    let mine: Vec<*mut PyObject> = operands.iter().copied().filter(|o| stand_in(*o)).collect();
    let at = |k: usize| operands.get(k).copied().unwrap_or(std::ptr::null_mut());
    let by_value = match ins.op {
        // The operator of the other operand's class may run, in C.
        Op::Binary(_) | Op::Compare(_) => !operands.iter().all(|o| defers(*o)),
        // `t.real`: a C getter of float.
        Op::LoadAttr(_) => stand_in(at(0)),
        // The value (args[0]) handed to a C setter of the object's class:
        // not a store into the instance's dict or a slot, nor one that runs
        // Python code of the class.
        Op::StoreAttr(_) => stand_in(at(0)) && !at(1).is_null() && tr_store_attr_kind(ffi::Py_TYPE(at(1)), name, 0, 0) == 2,
        Op::Bytecode { opcode, .. } => match opcode {
            // Builds and moves.
            opcodes::BUILD_LIST
            | opcodes::BUILD_SET
            | opcodes::BUILD_MAP
            | opcodes::BUILD_CONST_KEY_MAP
            | opcodes::BUILD_SLICE
            | opcodes::LIST_APPEND
            | opcodes::SET_ADD
            | opcodes::MAP_ADD
            | opcodes::LIST_EXTEND
            | opcodes::SET_UPDATE
            | opcodes::DICT_UPDATE
            | opcodes::DICT_MERGE
            | opcodes::UNPACK_SEQUENCE
            | opcodes::UNPACK_EX
            | opcodes::STORE_GLOBAL
            | opcodes::STORE_NAME
            | opcodes::IS_OP
            // The stand-in's own methods.
            | opcodes::FORMAT_VALUE
            | opcodes::UNARY_NEGATIVE
            | opcodes::UNARY_NOT
            | opcodes::CALL_INTRINSIC_1
            // A call is looked at with its callee (see handed).
            | opcodes::CALL_FUNCTION_EX => false,
            // An item of a builtin container, a key of one, a member of one.
            opcodes::STORE_SUBSCR | opcodes::STORE_SLICE => !holds_objects(at(1)),
            opcodes::BINARY_SUBSCR | opcodes::DELETE_SUBSCR | opcodes::BINARY_SLICE => !holds_objects(at(0)),
            opcodes::CONTAINS_OP => !holds_objects(at(1)),
            _ => true,
        },
        _ => false,
    };
    if by_value {
        read(&mine);
    }
}

// Whether a call of `callee` with these arguments (the receiver of a bound
// builtin method included) runs C code for the callee itself that is handed
// a stand-in: then `called` is to be given its result.
pub unsafe fn handed(callee: *mut PyObject, args: &[*mut PyObject]) -> bool {
    let python = ffi::PyFunction_Check(callee) != 0 || (ffi::Py_TYPE(callee) == &raw mut PyMethod_Type && ffi::PyFunction_Check(tr_bound_function(callee)) != 0) || !tr_type_function(callee, 0).is_null();
    if python {
        return false;
    }
    let mut found = vec![];
    find(tr_builtin_self(callee), 0, &mut found);
    args.iter().for_each(|a| find(*a, 0, &mut found));
    !found.is_empty()
}

// The call `handed` said yes to returned `result` (owned; NULL: it raised).
// `kw`: the names of the arguments given by keyword (the last ones), or
// NULL. What the instruction's result is (owned):
// torch._rust_inductor._c_call decides what the call means for the
// stand-ins, and may give a stand-in for the result.
pub unsafe fn called(callee: *mut PyObject, args: &[*mut PyObject], kw: *mut PyObject, result: *mut PyObject) -> *mut PyObject {
    let py = Python::assume_attached();
    let mut found = vec![];
    find(tr_builtin_self(callee), 0, &mut found);
    args.iter().for_each(|a| find(*a, 0, &mut found));
    if result.is_null() {
        // It raised: where the code goes from here depends on the value.
        let raised = ffi::PyErr_GetRaisedException();
        read(&found);
        ffi::PyErr_SetRaisedException(raised);
        return result;
    }
    let me = tr_builtin_self(callee);
    let all: Vec<Bound<'_, PyAny>> = (!me.is_null()).then_some(me).into_iter().chain(args.iter().copied()).filter(|a| !a.is_null()).map(|a| Bound::from_borrowed_ptr(py, a)).collect();
    let keywords = if kw.is_null() { 0 } else { ffi::PyTuple_GET_SIZE(kw) as usize };
    let Ok(all) = pyo3::types::PyTuple::new(py, all) else { return result };
    let Ok(keywords) = keywords.into_pyobject(py);
    let given = vec![Bound::from_borrowed_ptr(py, callee), all.into_any(), keywords.into_any(), Bound::from_borrowed_ptr(py, result)];
    match tell(py, "_c_call", given) {
        Some(new) => {
            ffi::Py_DECREF(result);
            new.into_ptr()
        }
        None => result,
    }
}
