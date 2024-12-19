#pragma once

#include <torch/csrc/utils/pythoncapi_compat.h>
namespace torch::xpu::python {

void initCommMethods(PyObject* module);

} // namespace torch::xpu::python
