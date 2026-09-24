#pragma once
#include <torch/csrc/python_headers.h>

namespace torch::symbolic {

void initSymbolicBindings(PyObject* module);

} // namespace torch::symbolic
