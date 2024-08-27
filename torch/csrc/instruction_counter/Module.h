#pragma once
#include <torch/csrc/python_headers.h>

namespace torch::instruction_counter {

void initModule(PyObject* module);

} // namespace torch::instruction_counter
