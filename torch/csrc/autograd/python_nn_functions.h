#pragma once

#include <torch/csrc/python_headers.h>
#include <pybind11/pybind11.h>

namespace torch { namespace autograd {

void initNNFunctions(PyObject* module);

}} // namespace torch::autograd
