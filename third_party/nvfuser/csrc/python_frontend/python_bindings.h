#pragma once

#include <torch/csrc/jit/python/pybind.h>
#include <torch/csrc/utils/pybind.h>

namespace nvfuser::python_frontend {
void initNvFuserPythonBindings(PyObject* module);
}
