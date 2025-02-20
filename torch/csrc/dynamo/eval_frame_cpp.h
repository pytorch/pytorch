#pragma once
#include <Python.h>
#include <stdbool.h>

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

#ifdef __cplusplus

} // extern "C"

#endif
