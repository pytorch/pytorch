#pragma once

#include <torch/csrc/utils/pybind.h>

namespace torch::jit {

void initJITBindings(PyObject* module);

} // namespace torch::jit
