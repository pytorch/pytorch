#pragma once

#include <torch/csrc/jit/python/pybind.h>

namespace torch {
namespace jit {
void initSPMDRuntimeBindings(PyObject* module);
} // namespace jit
} // namespace torch
