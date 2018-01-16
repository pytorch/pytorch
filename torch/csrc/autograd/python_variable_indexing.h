#pragma once

#include <Python.h>

namespace torch { namespace autograd {

Py_ssize_t THPVariable_length(PyObject* self);
PyObject* THPVariable_getitem(PyObject* self, PyObject* index);
int THPVariable_setitem(PyObject* self, PyObject* index, PyObject* value);

}} // namespace torch::autograd
