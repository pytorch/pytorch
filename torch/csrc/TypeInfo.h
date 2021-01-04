#pragma once

#include <torch/csrc/python_headers.h>

#include <ATen/ATen.h>

namespace torch {

void initTypeInfoBindings(PyObject* module);

} // namespace torch
