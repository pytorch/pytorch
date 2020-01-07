#pragma once

// @generated from tools/autograd/templates/python_nn_functions.h

#include <Python.h>
#include <pybind11/pybind11.h>

namespace torch { namespace autograd {

void initNNFunctions(PyObject* module);

}} // namespace torch::autograd
