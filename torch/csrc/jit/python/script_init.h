#pragma once

#include <torch/csrc/jit/python/pybind.h>

namespace torch {
namespace jit {
void initJitScriptBindings(PyObject* module, PyObject* parent);
} // namespace jit
} // namespace torch
