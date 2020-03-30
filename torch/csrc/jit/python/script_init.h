#pragma once

#include <torch/csrc/jit/python/pybind.h>

namespace torch {
namespace jit {
void initJitScriptBindings(PyObject* module);
} // namespace jit
} // namespace torch
