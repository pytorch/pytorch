#pragma once

#include <torch/csrc/jit/python/pybind.h>

namespace torch {
namespace jit {
namespace script {
void initJitScriptBindings(PyObject* module);
} // namespace script
} // namespace jit
} // namespace torch
