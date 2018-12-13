#pragma once

#include <torch/csrc/utils/pybind.h>

namespace torch { namespace onnx {

void initONNXBindings(PyObject* module);

}} // namespace torch::onnx
