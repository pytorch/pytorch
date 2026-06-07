#pragma once

// C2039 MSVC
#include <pybind11/complex.h>
#include <torch/csrc/utils/pybind.h>

#include <torch/csrc/utils/PythonWrapper.h>

namespace torch::dynamo {
void initDynamoBindings(PyObject* torch);
}
