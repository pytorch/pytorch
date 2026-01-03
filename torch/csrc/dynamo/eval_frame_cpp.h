#pragma once
#include <Python.h>

#include <torch/csrc/dynamo/eval_frame.h>
#include <torch/csrc/dynamo/extra_state.h>
#include <torch/csrc/dynamo/framelocals_mapping.h>
#ifdef __cplusplus

extern "C" {

#endif

PyObject* dynamo__custom_eval_frame(
    PyThreadState* tstate,
    THP_EVAL_API_FRAME_OBJECT* frame,
    int throw_flag,
    PyObject* callback);

PyObject* dynamo_set_code_exec_strategy(PyObject* dummy, PyObject* obj);
void dynamo_skip_code_recursive(PyCodeObject* code);

void dynamo_set_c_recursion_limit(int32_t limit);
int32_t dynamo_get_c_recursion_limit();

#ifdef __cplusplus

} // extern "C"

#endif
