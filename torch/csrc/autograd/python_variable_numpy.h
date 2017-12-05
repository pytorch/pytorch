#pragma once

#include <Python.h>

namespace torch { namespace autograd {

PyObject * THPVariable_numpy(PyObject* self, PyObject* arg);

}} // namespace torch::autograd
