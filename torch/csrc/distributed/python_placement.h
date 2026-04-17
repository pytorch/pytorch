#pragma once

#include <torch/csrc/utils/python_stub.h>

namespace torch::distributed {
void initPlacementBindings(PyObject* module);
} // namespace torch::distributed
