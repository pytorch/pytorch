#pragma once

#include "torch/csrc/autograd/function_hook.h"
#include "torch/csrc/utils/object_ptr.h"
#include <Python.h>

namespace torch { namespace autograd {

struct PyFunctionPreHook : public FunctionPreHook {
  PyFunctionPreHook(PyObject* dict, int grad_index);
  ~PyFunctionPreHook();
  variable_list operator()(const variable_list& grads) override;
  PyObject* dict;
  int grad_index;
};

struct PyFunctionPostHook : public FunctionPostHook {
  PyFunctionPostHook(PyObject* dict);
  ~PyFunctionPostHook();
  variable_list operator()(const variable_list& grad_input, const variable_list& grad_output) override;
  PyObject* dict;
};

}} // namespace torch::autograd
