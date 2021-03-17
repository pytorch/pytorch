#pragma once

#include <Python.h>

namespace torch {
namespace fx {
// Initialize Python bindings for Tensor Expressions
void initFx(PyObject* module);
} // namespace fx
} // namespace torch
