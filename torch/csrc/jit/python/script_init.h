#pragma once

#include <torch/csrc/jit/python/pybind.h>

namespace torch::jit {
void initJitScriptBindings(PyObject* module);
} // namespace torch::jit
