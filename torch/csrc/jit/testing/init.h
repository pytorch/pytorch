#pragma once

#include <torch/csrc/jit/pybind.h>

namespace torch {
namespace jit {
namespace testing {
void initJitTestingBindings(PyObject* module);

} // namespace testing
} // namespace jit
} // namespace torch
