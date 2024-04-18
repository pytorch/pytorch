#pragma once

#include <torch/csrc/autograd/function_hook.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>

namespace torch::dynamo::autograd {
class SwapSavedVariables;
} // namespace torch::dynamo::autograd

namespace torch::autograd {

struct PyFunctionTensorPreHook : public FunctionPreHook {
  PyFunctionTensorPreHook(PyObject* dict, size_t value_idx);
  ~PyFunctionTensorPreHook() override;
  variable_list operator()(const variable_list& values) override;
  void compiled_args(torch::dynamo::autograd::CompiledNodeArgs& args) override;
  PyObject* dict;
  size_t value_idx;
};

struct PyFunctionPreHook : public FunctionPreHook {
  PyFunctionPreHook(PyObject* dict);
  ~PyFunctionPreHook() override;
  variable_list operator()(const variable_list& values) override;
  void compiled_args(torch::dynamo::autograd::CompiledNodeArgs& args) override;
  PyObject* dict;
};

struct PyFunctionPostHook : public FunctionPostHook {
  PyFunctionPostHook(PyObject* dict);
  ~PyFunctionPostHook() override;
  variable_list operator()(
      const variable_list& outputs,
      const variable_list& inputs) override;
  void compiled_args(torch::dynamo::autograd::CompiledNodeArgs& args) override;
  PyObject* dict;
};

// PyFunctionTensorPostAccGradHooks is a dictionary of PostAccumulateGradHooks,
// and it is understandable if you are confused by why it's a subclass. We are
// simply following the precedent of PyFunctionPreHook and PyFunctionPostHook
// above to easily enroll into existing infrastructure.
struct PyFunctionTensorPostAccGradHooks : public PostAccumulateGradHook {
  PyFunctionTensorPostAccGradHooks(PyObject* dict);
  ~PyFunctionTensorPostAccGradHooks() override;
  void operator()(const Variable& tensor) override;
  void compiled_args(torch::dynamo::autograd::CompiledNodeArgs& args) override;
  void apply_with_saved(
      Variable& tensor,
      torch::dynamo::autograd::SwapSavedVariables& saved) override;
  PyObject* dict;
};

} // namespace torch::autograd
