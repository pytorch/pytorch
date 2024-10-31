#pragma once

#include <torch/csrc/jit/python/pybind.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace jit {
// Initialize Python bindings for JIT to_<backend> functions.
void initJitBackendBindings(PyObject* module);
} // namespace jit
} // namespace torch
