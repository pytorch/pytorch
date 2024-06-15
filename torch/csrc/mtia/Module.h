#pragma once

#include <torch/csrc/python_headers.h>

namespace torch {
namespace mtia {

// PyMethodDef* python_functions();
void initModule(PyObject* module);

} // namespace mtia
} // namespace torch
