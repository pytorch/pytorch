#pragma once
#include <torch/csrc/python_headers.h>

namespace torch::cpu {

void initModule(PyObject* module);

} // namespace torch::cpu
