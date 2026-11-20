// Sees the torch ops run during CFG execution at the PreDispatch key (before
// autograd), through the hook in csrc/op_hook.cpp. Each op is recorded as a
// side effect (see effects.rs) and, if asked for, printed to sys.stdout tagged
// with the frame and instruction running it.

use std::sync::atomic::{AtomicBool, Ordering};

use pyo3::prelude::*;

use super::ffi::{tr_current_frame, tr_frame_code, tr_frame_get_instr, PyCode_Addr2Line};
use crate::bindings::ffi::op_hook::{op_hook_disable, op_hook_enable, op_hook_pause};

static PRINT_TORCH_OPS: AtomicBool = AtomicBool::new(false);

/// Print every torch op (pre-autograd) run under call_with_cfg_execution.
#[pyfunction]
pub fn set_print_torch_ops(enabled: bool) {
    PRINT_TORCH_OPS.store(enabled, Ordering::Relaxed);
}

pub fn printing() -> bool {
    PRINT_TORCH_OPS.load(Ordering::Relaxed)
}

// Keeps the hook enabled on this thread for its lifetime; it runs only while
// effects are recorded (see effects::set_current) or ops printed.
pub struct OpHookGuard;

impl OpHookGuard {
    pub fn new() -> Self {
        op_hook_enable();
        if !super::effects::has_current() && !printing() {
            op_hook_pause(true);
        }
        OpHookGuard
    }
}

impl Drop for OpHookGuard {
    fn drop(&mut self) {
        op_hook_disable();
    }
}

// Called from the C++ hook, usually with the GIL released by the op's Python
// binding. An error becomes a C++ exception, which fails the op.
pub fn on_torch_op(op: &str, args: &str) -> Result<(), String> {
    super::effects::torch_op(op, args);
    if !PRINT_TORCH_OPS.load(Ordering::Relaxed) {
        return Ok(());
    }
    // Not Python::attach: the binding released the GIL behind pyo3's back, so
    // pyo3 still believes this thread holds it.
    unsafe {
        let gil = pyo3::ffi::PyGILState_Ensure();
        let py = Python::assume_attached();
        let result = location(py).and_then(|at| {
            let line = format!("torch op: {op}({args}){at}\n");
            py.import("sys")?.getattr("stdout")?.call_method1("write", (line,))?;
            Ok(())
        });
        let result = result.map_err(|e| e.to_string());
        pyo3::ffi::PyGILState_Release(gil);
        result
    }
}

// "  # qualname (file.py:line) @offset" for the innermost Python frame.
fn location(py: Python<'_>) -> PyResult<String> {
    unsafe {
        let f = tr_current_frame(pyo3::ffi::PyThreadState_Get());
        if f.is_null() {
            return Ok(String::new());
        }
        let unit = tr_frame_get_instr(f);
        let code = Bound::from_borrowed_ptr(py, tr_frame_code(f));
        let line = PyCode_Addr2Line(code.as_ptr(), unit * 2);
        let qualname: String = code.getattr("co_qualname")?.extract()?;
        let filename: String = code.getattr("co_filename")?.extract()?;
        let file = filename.rsplit('/').next().unwrap_or(&filename);
        Ok(format!("  # {qualname} ({file}:{line}) @{}", unit * 2))
    }
}
