#pragma once
#include <torch/csrc/python_headers.h>

namespace torch {
  // Sometimes we don't want infinite recursion for subclasses,
  // Or a way to achieve the old behaviour.

  // This is an internal utility, not exposed to users.
  bool torch_function_enabled();
  PyObject* disabled_torch_function_impl();
  void set_disabled_torch_function_impl(PyObject* value);
  bool check_has_torch_function(PyObject* obj);
}

PyObject* THPModule_isEnabledTorchFunction(PyObject* self, PyObject *unused);
PyObject* THPModule_DisableTorchFunctionType();
PyObject* THPModule_disable_torch_function(PyObject *self, PyObject *args);
PyObject* THPModule_has_torch_function(PyObject*, PyObject *arg);
PyObject* THPModule_has_torch_function_unary(PyObject*, PyObject *obj);
PyObject* THPModule_has_torch_function_variadic(PyObject*, PyObject *const *args, Py_ssize_t nargs);
