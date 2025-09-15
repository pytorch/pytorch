#pragma once

#include <torch/csrc/python_headers.h>

namespace torch::jit {

void initTreeViewBindings(PyObject* module);

} // namespace torch::jit
