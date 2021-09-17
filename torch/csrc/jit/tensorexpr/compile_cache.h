#pragma once

#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace jit {
namespace tensorexpr {

/// Initialize python bindings for kernel compilation cache.
void initTensorExprCompileCacheBindings(PyObject* teModule);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
