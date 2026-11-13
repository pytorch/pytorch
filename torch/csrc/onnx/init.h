#pragma once

#include <torch/csrc/utils/pybind.h>

namespace torch::onnx {

void initONNXBindings(PyObject* module);

} // namespace torch::onnx
