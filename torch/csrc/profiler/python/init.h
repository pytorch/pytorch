#pragma once

#include <torch/csrc/utils/python_stub.h>

namespace torch {
namespace profiler {

void initPythonBindings(PyObject* module);

} // namespace profiler
} // namespace torch
