#pragma once

#include <torch/csrc/python_headers.h>

namespace torch {

void initTensorIteratorBindings(PyObject* module);

} // namespace torch
