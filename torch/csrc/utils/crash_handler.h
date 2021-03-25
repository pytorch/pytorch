#pragma once
#include <pybind11/pybind11.h>
#include <torch/csrc/jit/python/pybind_utils.h>



namespace torch {
namespace crash_handler {

void initCrashHandlerBindings(PyObject* module);


} // namespace crash_handler
} // namepsace torch
