#pragma once
#include <stdbool.h>

#include <torch/csrc/dynamo/extra_state.h>
#include <torch/csrc/utils/python_compat.h>
#ifdef __cplusplus

extern "C" {

PyObject* torch_c_dynamo_eval_frame_init(void);

#endif

#if IS_PYTHON_3_12_PLUS
extern const size_t sys_monitoring_num_callables;
PyObject** get_monitoring_callables(PyInterpreterState* interp);
#endif

// All the eval APIs change in 3.11 so we need to decide which one to use on the
// fly https://docs.python.org/3/c-api/init.html#c._PyFrameEvalFunction
#if IS_PYTHON_3_11_PLUS
#define THP_EVAL_API_FRAME_OBJECT _PyInterpreterFrame
#else
#define THP_EVAL_API_FRAME_OBJECT PyFrameObject
#endif // IS_PYTHON_3_11_PLUS

// We need to be able to return the _PyInterpreterFrame to python so create
// a python binding for it

typedef struct THPPyInterpreterFrame {
  PyObject_HEAD
  THP_EVAL_API_FRAME_OBJECT* frame; // Borrowed reference
  PyObject* locals;
} THPPyInterpreterFrame;

THPPyInterpreterFrame* THPPyInterpreterFrame_New(
    THP_EVAL_API_FRAME_OBJECT* frame);

extern bool is_skip_guard_eval_unsafe;

void clear_old_frame_if_python_312_plus(
    PyThreadState* tstate,
    THP_EVAL_API_FRAME_OBJECT* frame);

void eval_frame_callback_set(PyObject* obj);

const char* get_frame_name(THP_EVAL_API_FRAME_OBJECT* frame);

PyObject* dynamo_eval_frame_default(
    PyThreadState* tstate,
    THP_EVAL_API_FRAME_OBJECT* frame,
    int throw_flag);

PyObject* dynamo_eval_custom_code(
    PyThreadState* tstate,
    THP_EVAL_API_FRAME_OBJECT* frame,
    PyCodeObject* code,
    const char* trace_annotation,
    int throw_flag);

#ifdef __cplusplus

} // extern "C"

#endif
