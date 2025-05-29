#pragma once

#include <torch/csrc/python_headers.h>

namespace torch::vulkan {

PyMethodDef* python_functions();
void initModule(PyObject* module);

} // namespace torch::vulkan
