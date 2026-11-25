// Declarations for csrc/frame_shim.c, csrc/exec_ops.c and the CPython APIs
// pyo3::ffi doesn't cover.

use std::os::raw::{c_int, c_void};

use pyo3::ffi::{PyObject, PyThreadState};

#[repr(C)]
pub struct Frame {
    _private: [u8; 0],
}

pub type EvalFrameFn = unsafe extern "C" fn(*mut PyThreadState, *mut Frame, c_int) -> *mut PyObject;

unsafe extern "C" {
    // frame_shim.c
    pub fn tr_frame_code(f: *mut Frame) -> *mut PyObject;
    pub fn tr_frame_qualname(f: *mut Frame) -> *mut PyObject;
    pub fn tr_frame_localsplus(f: *mut Frame) -> *mut *mut PyObject;
    pub fn tr_frame_func(f: *mut Frame) -> *mut PyObject;
    pub fn tr_frame_owned_by_generator(f: *mut Frame) -> c_int;
    pub fn tr_frame_prev_instr_slot(f: *mut Frame) -> *mut *mut u16;
    pub fn tr_frame_code_units(f: *mut Frame) -> *mut u16;
    pub fn tr_frame_get_instr(f: *mut Frame) -> c_int;
    pub fn tr_frame_nlocalsplus(f: *mut Frame) -> c_int;
    pub fn tr_frame_enter(t: *mut PyThreadState, f: *mut Frame) -> c_int;
    pub fn tr_frame_exit(t: *mut PyThreadState, f: *mut Frame);
    pub fn tr_frame_clear_and_pop(t: *mut PyThreadState, f: *mut Frame);
    pub fn tr_traceback_here(f: *mut Frame) -> c_int;
    pub fn tr_current_frame(t: *mut PyThreadState) -> *mut Frame;
    pub fn tr_tracing_active(t: *mut PyThreadState) -> c_int;
    pub fn tr_frame_push(f: *mut Frame, obj: *mut PyObject);
    pub fn tr_frame_globals(f: *mut Frame) -> *mut PyObject;
    pub fn tr_frame_builtins(f: *mut Frame) -> *mut PyObject;
    pub fn tr_frame_pop(f: *mut Frame) -> *mut PyObject;
    pub fn tr_frame_stack_depth(f: *mut Frame) -> c_int;
    pub fn tr_gen_suspend(t: *mut PyThreadState, f: *mut Frame, unit: c_int);
    pub fn tr_gen_exit(t: *mut PyThreadState, f: *mut Frame);
    pub fn tr_code_localsplusnames(code: *mut PyObject) -> *mut PyObject;
    pub fn tr_eval_breaker_flag(t: *mut PyThreadState) -> *mut c_int;

    // exec_ops.c
    pub fn tr_unbound_local(f: *mut Frame, slot: c_int);
    pub fn tr_unbound_deref(f: *mut Frame, slot: c_int);
    pub fn tr_load_global(f: *mut Frame, namei: c_int, cache: *mut GlobalCache) -> *mut PyObject;
    pub fn tr_load_method(name: *mut PyObject, owner: *mut PyObject, out: *mut *mut PyObject) -> c_int;
    pub fn tr_bound_function(callable: *mut PyObject) -> *mut PyObject;
    pub fn tr_partial_func(callable: *mut PyObject) -> *mut PyObject;
    pub fn tr_type_function(obj: *mut PyObject, getattr: c_int) -> *mut PyObject;
    pub static mut PyMethod_Type: pyo3::ffi::PyTypeObject;
    pub fn tr_type_attr_is(obj: *mut PyObject, name: *mut PyObject, f: *mut PyObject) -> c_int;
    pub fn tr_getattr_falls_back(obj: *mut PyObject, name: *mut PyObject, f: *mut PyObject) -> c_int;
    pub fn tr_call(args: *const *mut PyObject, argc: c_int, kwnames: *mut PyObject) -> *mut PyObject;
    pub fn tr_binary_op(op: c_int, lhs: *mut PyObject, rhs: *mut PyObject) -> *mut PyObject;
    pub fn tr_make_function(f: *mut Frame, flags: c_int, args: *const *mut PyObject) -> *mut PyObject;
    pub fn tr_load_from_dict_or_deref(f: *mut Frame, class_dict: *mut PyObject, slot: c_int) -> *mut PyObject;
    pub fn tr_do_raise(exc: *mut PyObject, cause: *mut PyObject) -> c_int;
    pub fn tr_send(receiver: *mut PyObject, v: *mut PyObject, out: *mut *mut PyObject) -> c_int;
    pub fn tr_generic(
        f: *mut Frame,
        opcode: c_int,
        oparg: c_int,
        args: *const *mut PyObject,
        out: *mut *mut PyObject,
    ) -> c_int;

    pub fn tr_attr_home(obj: *mut PyObject, name: *mut PyObject, dict: *mut *mut PyObject, generic: c_int) -> c_int;
    pub fn tr_type_version(tp: *mut pyo3::ffi::PyTypeObject) -> u32;
    pub fn tr_type_version_is(tp: *mut pyo3::ffi::PyTypeObject, tag: u32) -> c_int;
    pub fn tr_type_immutable(tp: *mut pyo3::ffi::PyTypeObject) -> c_int;
    pub fn tr_type_lookup(tp: *mut pyo3::ffi::PyTypeObject, name: *mut PyObject) -> *mut PyObject;
    pub fn tr_type_heap(tp: *mut pyo3::ffi::PyTypeObject) -> c_int;
    pub fn tr_super_next(cls: *mut PyObject, obj: *mut PyObject) -> *mut PyObject;
    pub fn tr_dict_version(d: *mut PyObject) -> u64;
    pub fn tr_view_dict(o: *mut PyObject) -> *mut PyObject;
    pub fn tr_proxy_dict(o: *mut PyObject) -> *mut PyObject;
    pub fn tr_builtin_self(callable: *mut PyObject) -> *mut PyObject;
    pub fn tr_calls_builtin(callable: *mut PyObject) -> c_int;

    // CPython
    pub fn _PySet_NextEntry(set: *mut PyObject, pos: *mut pyo3::ffi::Py_ssize_t, key: *mut *mut PyObject, hash: *mut pyo3::ffi::Py_hash_t) -> c_int;
    pub fn _PyInterpreterState_GetEvalFrameFunc(interp: *mut pyo3::ffi::PyInterpreterState) -> EvalFrameFn;
    pub fn _PyInterpreterState_SetEvalFrameFunc(interp: *mut pyo3::ffi::PyInterpreterState, f: EvalFrameFn);
    pub fn PyUnstable_Eval_RequestCodeExtraIndex(free: unsafe extern "C" fn(*mut c_void)) -> isize;
    pub fn PyUnstable_Code_GetExtra(code: *mut PyObject, index: isize, extra: *mut *mut c_void) -> c_int;
    pub fn PyUnstable_Code_SetExtra(code: *mut PyObject, index: isize, extra: *mut c_void) -> c_int;
    pub fn PyFunction_GetClosure(func: *mut PyObject) -> *mut PyObject;
    pub fn PyCell_New(obj: *mut PyObject) -> *mut PyObject;
    pub fn PyCell_Get(cell: *mut PyObject) -> *mut PyObject;
    pub fn PyCell_Set(cell: *mut PyObject, value: *mut PyObject) -> c_int;
    pub fn PyCode_Addr2Line(code: *mut PyObject, byte_offset: c_int) -> c_int;
}

// tr_global_cache in exec_ops.c.
#[repr(C)]
pub struct GlobalCache {
    globals: *mut PyObject,
    globals_version: u64,
    builtins: *mut PyObject,
    builtins_version: u64,
    value: *mut PyObject,
}

impl Default for GlobalCache {
    fn default() -> Self {
        GlobalCache {
            globals: std::ptr::null_mut(),
            globals_version: 0,
            builtins: std::ptr::null_mut(),
            builtins_version: 0,
            value: std::ptr::null_mut(),
        }
    }
}

// Per-Cfg executor state written while running, which only happens with the
// GIL held, so a shared Prepared may hand out mutable access.
pub struct GilCell<T>(std::cell::UnsafeCell<T>);

unsafe impl<T> Sync for GilCell<T> {}
unsafe impl<T> Send for GilCell<T> {}

impl<T> GilCell<T> {
    pub fn new(t: T) -> Self {
        GilCell(std::cell::UnsafeCell::new(t))
    }

    pub fn get(&self) -> *mut T {
        self.0.get()
    }
}
