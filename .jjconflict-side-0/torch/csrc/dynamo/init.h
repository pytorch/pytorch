#pragma once

// C2039 MSVC
#include <pybind11/complex.h>
#include <torch/csrc/utils/pybind.h>

#include <Python.h>

namespace torch::dynamo {
void initDynamoBindings(PyObject* torch);
}
