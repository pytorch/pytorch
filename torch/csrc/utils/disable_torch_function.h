#pragma once
#include <torch/csrc/python_headers.h>

namespace torch {
  // Sometimes we don't want infinite recursion for subclasses,
  // Or a way to achieve the old behaviour.

  // This is an internal utility, not exposed to users.
  bool torch_function_enabled();
}

PyObject* THPModule_isEnabledTorchFunction(PyObject* self, PyObject *unused);
PyObject* THPModule_DisableTorchFunctionType();
