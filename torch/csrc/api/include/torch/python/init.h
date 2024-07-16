#pragma once

#include <torch/csrc/utils/python_stub.h>

namespace torch {
namespace python {
/// Initializes Python bindings for the C++ frontend.
void init_bindings(PyObject* module);
} // namespace python
} // namespace torch
