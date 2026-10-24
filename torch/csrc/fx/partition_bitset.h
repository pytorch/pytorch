#pragma once

#include <torch/csrc/python_headers.h>

namespace torch::fx {

bool initPartitionBitsetBindings(PyObject* module);

} // namespace torch::fx
