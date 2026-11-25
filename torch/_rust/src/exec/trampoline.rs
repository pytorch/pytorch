// A few intrinsics (the PEP 695 typing ones) call CPython internals that
// aren't exported, and their Python-level equivalents differ observably (e.g.
// TypeVar.__module__). They run through a one-instruction trampoline code
// object evaluated by CPython's own loop; anything the intrinsic calls back
// into still runs through the executor.

use std::collections::HashMap;
use std::os::raw::c_void;
use std::sync::OnceLock;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::cfg::opcodes;

use super::ffi::PyUnstable_Code_SetExtra;

// co_extra value marking trampoline code; the hook passes such frames to
// CPython's evaluator.
static MARK: u8 = 0;

pub fn mark() -> *mut c_void {
    &MARK as *const u8 as *mut c_void
}

pub fn needed(opcode: u8, arg: u32) -> bool {
    match opcode {
        // TYPEVAR, PARAMSPEC, TYPEVARTUPLE, SUBSCRIPT_GENERIC, TYPEALIAS
        opcodes::CALL_INTRINSIC_1 => (7..=11).contains(&arg),
        // TYPEVAR_WITH_BOUND, TYPEVAR_WITH_CONSTRAINTS
        opcodes::CALL_INTRINSIC_2 => (2..=3).contains(&arg),
        _ => false,
    }
}

// Built once, when CFG execution is first turned on (not on first use: building
// calls compile(), whose audit hook may itself be Python code run by the
// executor).
struct Trampolines {
    // By (opcode, oparg).
    intrinsics: HashMap<(u8, u32), Py<PyAny>>,
    breaker: Py<PyAny>,
}

static TRAMPOLINES: OnceLock<Trampolines> = OnceLock::new();

pub fn init(py: Python<'_>) -> PyResult<()> {
    if TRAMPOLINES.get().is_some() {
        return Ok(());
    }
    let mut intrinsics = HashMap::new();
    for (opcode, args) in [(opcodes::CALL_INTRINSIC_1, 7..=11), (opcodes::CALL_INTRINSIC_2, 2..=3)] {
        for arg in args {
            intrinsics.insert((opcode, arg), build(py, Some((opcode, arg)))?);
        }
    }
    let breaker = build(py, None)?;
    let _ = TRAMPOLINES.set(Trampolines { intrinsics, breaker });
    Ok(())
}

// `def t(a): return -a` (or `a is b`) with the UNARY_NEGATIVE (IS_OP)
// replaced by the intrinsic (neither has inline caches), or for None an empty
// function.
fn build(py: Python<'_>, intrinsic: Option<(u8, u32)>) -> PyResult<Py<PyAny>> {
    let src = match intrinsic {
        None => "def t():\n    pass\n",
        Some((opcodes::CALL_INTRINSIC_1, _)) => "def t(a):\n    return -a\n",
        Some(_) => "def t(a, b):\n    return a is b\n",
    };
    let builtins = py.import("builtins")?;
    let module = builtins.call_method1("compile", (src, "<cfg trampoline>", "exec"))?;
    let mut code = module.getattr("co_consts")?.get_item(0)?;
    if let Some((opcode, arg)) = intrinsic {
        let placeholder = if opcode == opcodes::CALL_INTRINSIC_1 {
            opcodes::UNARY_NEGATIVE
        } else {
            opcodes::IS_OP
        };
        let mut bytes = code.getattr("co_code")?.cast_into::<PyBytes>()?.as_bytes().to_vec();
        let pos = (0..bytes.len())
            .step_by(2)
            .find(|i| bytes[*i] == placeholder)
            .expect("trampoline template has the placeholder");
        bytes[pos] = opcode;
        bytes[pos + 1] = u8::try_from(arg).expect("intrinsic oparg fits a byte");
        let kwargs = PyDict::new(py);
        kwargs.set_item("co_code", PyBytes::new(py, &bytes))?;
        code = code.call_method("replace", (), Some(&kwargs))?;
    }
    unsafe {
        if PyUnstable_Code_SetExtra(code.as_ptr(), super::extra_index(), mark()) < 0 {
            return Err(PyErr::fetch(py));
        }
    }
    let globals = PyDict::new(py);
    globals.set_item("__builtins__", builtins)?;
    Ok(py.import("types")?.getattr("FunctionType")?.call1((code, globals))?.unbind())
}

fn trampolines() -> &'static Trampolines {
    TRAMPOLINES.get().expect("trampoline::init runs before CFG execution starts")
}

// CPython runs its eval breaker (_Py_HandlePending: signals, pending calls,
// scheduled GC, GIL drop requests, async exceptions; not exported) at the
// RESUME starting each function, so calling an empty function through its
// loop runs exactly that. Returns false with an exception set if a handler
// raised.
pub unsafe fn eval_breaker() -> bool {
    Python::attach(|py| match trampolines().breaker.bind(py).call0() {
        Ok(_) => true,
        Err(e) => {
            e.restore(py);
            false
        }
    })
}

// Runs the intrinsic on `args`; NULL with an exception set on failure.
pub unsafe fn call(opcode: u8, arg: u32, args: &[*mut pyo3::ffi::PyObject]) -> *mut pyo3::ffi::PyObject {
    Python::attach(|py| {
        let func = trampolines().intrinsics[&(opcode, arg)].bind(py);
        let args: Vec<Bound<'_, PyAny>> = args.iter().map(|a| Bound::from_borrowed_ptr(py, *a)).collect();
        match pyo3::types::PyTuple::new(py, args).and_then(|t| func.call1(t)) {
            Ok(obj) => obj.into_ptr(),
            Err(e) => {
                e.restore(py);
                std::ptr::null_mut()
            }
        }
    })
}
