#pragma once

// #include <c10/core/Device.h>
// #include <c10/core/DispatchKey.h>
// #include <c10/core/ScalarType.h>
#include <torch/csrc/python_headers.h>

// namespace at {
// class Tensor;
// } // namespace at

namespace torch::fake_tensor {

void initialize(PyObject* module);

} // namespace torch::fake_tensor
