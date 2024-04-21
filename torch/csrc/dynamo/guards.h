#pragma once
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>

PyObject* torch_c_dynamo_guards_init();

// interfaces for extra_state and eval_frame.c because RootGuardManager class is
// not visible there.
void* convert_to_root_guard_manager(py::object root);
bool run_root_guard_manager(void* root, PyObject* f_locals);
