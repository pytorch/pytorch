#pragma once

// C2039 MSVC
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>

#include <Python.h>

namespace torch {
namespace dynamo {
void initDynamoBindings(PyObject* torch);
}
} // namespace torch
