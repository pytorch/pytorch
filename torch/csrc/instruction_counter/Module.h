#pragma once
#include <torch/csrc/python_headers.h>

namespace torch {
namespace instruction_counter {

void initModule(PyObject* module);

} // namespace instruction_counter
} // namespace torch
