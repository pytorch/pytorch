#pragma once

#include <torch/csrc/jit/python/pybind.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace jit {
// Initialize Python bindings for Tensor Expressions
void initTensorExprBindings(PyObject* module);
} // namespace jit
} // namespace torch
