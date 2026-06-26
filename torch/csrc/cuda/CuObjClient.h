#pragma once

#include <torch/csrc/python_headers.h>

namespace torch::cuda::shared {

void initCuObjBindings(PyObject* module);

} // namespace torch::cuda::shared
