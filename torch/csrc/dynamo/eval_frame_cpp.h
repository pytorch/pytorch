#pragma once
#include <Python.h>

#include <torch/csrc/dynamo/eval_frame.h>
#include <torch/csrc/dynamo/extra_state.h>
#include <torch/csrc/dynamo/framelocals_mapping.h>

#ifdef __cplusplus

#include <torch/csrc/utils/pybind.h>

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

// Bytecode debugger callback functions
void set_bytecode_debugger_callback(py::object callback);
py::object get_bytecode_debugger_callback();

// NullStackValue - sentinel class for representing NULL values on Python stack
class NullStackValue {
 public:
  static NullStackValue& get_singleton();
};

py::object get_null_stack_value();
py::list _get_frame_value_stack_at_depth(
    const py::handle& frame_obj,
    int depth);

#endif
