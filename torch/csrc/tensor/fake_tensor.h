#pragma once

#include <torch/csrc/utils/python_stub.h>

namespace torch::fake_tensor {

void initialize(PyObject* module);

} // namespace torch::fake_tensor
