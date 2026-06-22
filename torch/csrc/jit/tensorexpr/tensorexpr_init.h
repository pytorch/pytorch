#pragma once

#include <torch/csrc/jit/python/pybind.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::jit {
// Initialize Python bindings for Tensor Expressions
void initTensorExprBindings(PyObject* module);
} // namespace torch::jit
