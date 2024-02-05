#pragma once

// C2039 MSVC
#include <torch/csrc/utils/pybind.h>
#include <pybind11/complex.h>

#include <Python.h>

namespace torch {
namespace dynamo {
void initDynamoBindings(PyObject* torch);
}
} // namespace torch
