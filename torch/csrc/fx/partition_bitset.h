#pragma once

#include <torch/csrc/python_headers.h>

namespace torch::fx {

void initPartitionBitsetBindings(PyObject* module);

} // namespace torch::fx
