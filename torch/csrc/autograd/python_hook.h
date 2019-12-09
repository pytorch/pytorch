#pragma once

#include <torch/csrc/python_headers.h>
#include <torch/csrc/autograd/function_hook.h>
#include <torch/csrc/utils/object_ptr.h>

namespace torch { namespace autograd {

struct PyFunctionPreHook : public FunctionPreHook {
  PyFunctionPreHook(PyObject* dict, int value_idx);
  ~PyFunctionPreHook() override;
  variable_list operator()(const variable_list& values) override;
  PyObject* dict;
  int value_idx;
};

struct PyFunctionPostHook : public FunctionPostHook {
  PyFunctionPostHook(PyObject* dict);
  ~PyFunctionPostHook() override;
  variable_list operator()(const variable_list& outputs, const variable_list& inputs) override;
  PyObject* dict;
};

}} // namespace torch::autograd
