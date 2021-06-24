#pragma once

#include <torch/csrc/jit/python/pybind.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace jit {
void initTensorExprCompileCacheBindings(PyObject* module);
} // namespace jit
} // namespace torch
