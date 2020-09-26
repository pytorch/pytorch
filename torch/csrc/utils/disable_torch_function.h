#pragma once
#include <torch/csrc/python_headers.h>

namespace torch {
  // Sometimes we don't want infinite recursion for subclasses,
  // Or a way to achieve the old behaviour.

  // This is an internal utility, not exposed to users.
  bool torch_function_enabled();
  PyObject* disabled_torch_function_impl();
  void set_disabled_torch_function_impl(PyObject* value);
}

PyObject* THPModule_isEnabledTorchFunction(PyObject* self, PyObject *unused);
PyObject* THPModule_DisableTorchFunctionType();
PyObject* THPModule_disable_torch_function(PyObject *self, PyObject *args);