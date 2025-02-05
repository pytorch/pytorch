#pragma once
#include <Python.h>
#include <stdbool.h>

#include <torch/csrc/dynamo/extra_state.h>
#include <torch/csrc/dynamo/framelocals_mapping.h>
#ifdef __cplusplus

extern "C" {

PyObject* torch_c_dynamo_eval_frame_init(void);

#endif

// All the eval APIs change in 3.11 so we need to decide which one to use on the
// fly https://docs.python.org/3/c-api/init.html#c._PyFrameEvalFunction
#if IS_PYTHON_3_11_PLUS
#define THP_EVAL_API_FRAME_OBJECT _PyInterpreterFrame
#else
#define THP_EVAL_API_FRAME_OBJECT PyFrameObject
#endif // IS_PYTHON_3_11_PLUS

extern PyObject* skip_code_recursive_flag;
extern PyObject* cache_limit_hit_flag;
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

PyObject* dynamo_call_callback(
    PyObject* callable,
    THP_EVAL_API_FRAME_OBJECT* _frame,
    FrameLocalsMapping* locals,
    CacheEntry* cache_entry,
    FrameState* frame_state);

#ifdef __cplusplus

} // extern "C"

#endif
