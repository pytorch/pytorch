#pragma once
#include <torch/csrc/python_headers.h>

namespace torch {
namespace cpu {

void initModule(PyObject* module);

} // namespace cpu
} // namespace torch
