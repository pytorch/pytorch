#pragma once

#include <torch/csrc/autograd/function_hook.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>

namespace torch {
namespace autograd {

struct PyFunctionTensorPreHook : public FunctionPreHook {
  PyFunctionTensorPreHook(PyObject* dict, int value_idx);
  ~PyFunctionTensorPreHook() override;
  variable_list operator()(const variable_list& values) override;
  void compiled_args(CompiledNodeArgs& args) override;
  PyObject* dict;
  int value_idx;
};

struct PyFunctionPreHook : public FunctionPreHook {
  PyFunctionPreHook(PyObject* dict);
  ~PyFunctionPreHook() override;
  variable_list operator()(const variable_list& values) override;
  void compiled_args(CompiledNodeArgs& args) override;
  PyObject* dict;
};

struct PyFunctionPostHook : public FunctionPostHook {
  PyFunctionPostHook(PyObject* dict);
  ~PyFunctionPostHook() override;
  variable_list operator()(
      const variable_list& outputs,
      const variable_list& inputs) override;
  void compiled_args(CompiledNodeArgs& args) override;
  PyObject* dict;
};

} // namespace autograd
} // namespace torch
