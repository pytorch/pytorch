#pragma once

#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/python_headers.h>

namespace torch {
namespace autograd {

Py_ssize_t THPVariable_length(PyObject* self);
PyObject* THPVariable_getitem(PyObject* self, PyObject* index);
int THPVariable_setitem(PyObject* self, PyObject* index, PyObject* value);

Variable valueToTensor(
    c10::TensorOptions options,
    PyObject* value,
    const at::Device& device);

} // namespace autograd
} // namespace torch
