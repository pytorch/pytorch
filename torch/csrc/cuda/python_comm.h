#pragma once

#include <torch/csrc/utils/pythoncapi_compat.h>
namespace torch::cuda::python {

void initCommMethods(PyObject* module);

} // namespace torch::cuda::python
