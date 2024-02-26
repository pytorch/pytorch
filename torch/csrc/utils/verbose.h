#pragma once
#include <torch/csrc/python_headers.h>

namespace torch {

void initVerboseBindings(PyObject* module);

} // namespace torch
