#pragma once
#include <torch/csrc/python_headers.h>

namespace torch {
  bool torch_function_enabled();
}

PyObject* THPModule_isEnabledTorchFunction(PyObject* self, PyObject *unused);
PyObject* THPModule_DisableTorchFunctionType();
