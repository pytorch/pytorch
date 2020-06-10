#pragma once

#include <Python.h>
#include <pybind11/pybind11.h>

namespace torch { namespace autograd {

void initFFTFunctions(PyObject* module);

}} // namespace torch::autograd
