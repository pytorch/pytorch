// A PEP 523 eval-frame hook that prints the bytecode of every Python function
// evaluated while `call_with_bytecode_printing` is on the stack. Requires
// CPython 3.12+ (PyUnstable_InterpreterFrame_GetCode).

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::os::raw::c_int;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::exec::ffi::{
    EvalFrameFn, Frame as InterpreterFrame, _PyInterpreterState_GetEvalFrameFunc,
    _PyInterpreterState_SetEvalFrameFunc,
};

extern "C" {
    fn PyUnstable_InterpreterFrame_GetCode(frame: *mut InterpreterFrame) -> *mut ffi::PyObject;
}

// The eval-frame function is per-interpreter but printing is per-thread, so the
// hook stays installed while any thread is printing and other threads fall
// straight through to whatever eval-frame function was installed before us.
// INSTALLED counts active `PrintingGuard`s across all threads; PREV_EVAL_FRAME
// is read lock-free on every frame.
static INSTALLED: Mutex<usize> = Mutex::new(0);
static PREV_EVAL_FRAME: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static DEPTH: Cell<usize> = const { Cell::new(0) };
    static IN_HOOK: Cell<bool> = const { Cell::new(false) };
    // Code objects already printed during the current outermost call. Holding a
    // strong reference keeps the address from being reused by a new code object.
    static SEEN: RefCell<HashMap<usize, Py<PyAny>>> = RefCell::new(HashMap::new());
}

unsafe extern "C" fn eval_frame(
    tstate: *mut ffi::PyThreadState,
    frame: *mut InterpreterFrame,
    throwflag: c_int,
) -> *mut ffi::PyObject {
    // With throwflag set an exception is pending, so we must not run Python
    // code; the frame was already printed when it was first entered anyway.
    if throwflag == 0 && DEPTH.get() > 0 && !IN_HOOK.get() {
        IN_HOOK.set(true);
        print_frame_code(frame);
        IN_HOOK.set(false);
    }
    let prev: EvalFrameFn = std::mem::transmute(PREV_EVAL_FRAME.load(Ordering::Acquire));
    prev(tstate, frame, throwflag)
}

unsafe fn print_frame_code(frame: *mut InterpreterFrame) {
    Python::attach(|py| {
        let code = Bound::from_owned_ptr(py, PyUnstable_InterpreterFrame_GetCode(frame));
        let first_time = SEEN.with_borrow_mut(|seen| {
            seen.insert(code.as_ptr() as usize, code.clone().unbind())
                .is_none()
        });
        if first_time {
            if let Err(err) = print_code(py, &code) {
                err.write_unraisable(py, Some(&code));
            }
        }
    })
}

fn print_code(py: Python<'_>, code: &Bound<'_, PyAny>) -> PyResult<()> {
    let header = format!(
        "=== {} ({}:{}) ===",
        code.getattr("co_qualname")?,
        code.getattr("co_filename")?,
        code.getattr("co_firstlineno")?,
    );
    py.import("builtins")?.call_method1("print", (header,))?;
    // depth=0: nested code objects (inner functions, lambdas) are printed when
    // they actually run, not when their enclosing function is printed.
    let kwargs = PyDict::new(py);
    kwargs.set_item("depth", 0)?;
    py.import("dis")?.call_method("dis", (code,), Some(&kwargs))?;
    let cfg_text = match crate::cfg::python::code_info(code)
        .and_then(|info| crate::cfg::build(info, true, None).map_err(crate::cfg::python::to_pyerr))
    {
        Ok(cfg) => format!("--- CFG ---\n{}", cfg.render(false, false)),
        Err(err) => format!("<cfg unsupported: {err}>"),
    };
    py.import("builtins")?.call_method1("print", (cfg_text.trim_end(),))?;
    Ok(())
}

struct PrintingGuard;

impl PrintingGuard {
    fn new() -> Self {
        if DEPTH.get() == 0 {
            SEEN.with_borrow_mut(HashMap::clear);
        }
        DEPTH.set(DEPTH.get() + 1);
        let mut installed = INSTALLED.lock().unwrap();
        if *installed == 0 {
            unsafe {
                let interp = ffi::PyInterpreterState_Get();
                let prev = _PyInterpreterState_GetEvalFrameFunc(interp);
                PREV_EVAL_FRAME.store(prev as usize, Ordering::Release);
                _PyInterpreterState_SetEvalFrameFunc(interp, eval_frame);
            }
        }
        *installed += 1;
        PrintingGuard
    }
}

impl Drop for PrintingGuard {
    fn drop(&mut self) {
        let mut installed = INSTALLED.lock().unwrap();
        *installed -= 1;
        if *installed == 0 {
            unsafe {
                let prev: EvalFrameFn =
                    std::mem::transmute(PREV_EVAL_FRAME.load(Ordering::Acquire));
                _PyInterpreterState_SetEvalFrameFunc(ffi::PyInterpreterState_Get(), prev);
            }
        }
        drop(installed);
        DEPTH.set(DEPTH.get() - 1);
        if DEPTH.get() == 0 {
            SEEN.with_borrow_mut(HashMap::clear);
        }
    }
}

/// Call `func(*args, **kwargs)`, printing the bytecode of each Python function
/// evaluated on this thread during the call (each code object once).
#[pyfunction]
#[pyo3(signature = (func, /, *args, **kwargs))]
pub fn call_with_bytecode_printing<'py>(
    func: &Bound<'py, PyAny>,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let _guard = PrintingGuard::new();
    func.call(args, kwargs)
}
