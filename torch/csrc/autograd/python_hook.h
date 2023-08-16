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
  void compiled_args(torch::dynamo::autograd::CompiledNodeArgs& args) override;
  PyObject* dict;
  int value_idx;
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

// PyFunctionTensorPostAccGradHook is a DICTIONARY of PostAccumulateGradHooks,
// NOT just one hook! Thus, adding one hook is actually just replacing the
// existing dictionary with a (bigger) dictionary of hooks.
//
// Why? This way, we can take advantage of the existing infra for the other
// hooks (like calling the _call_hooks endpoint). I suppose a more adequate
// name would be PyFunctionTensorPostAccGradHooks but let's follow the
// precedent set by the hooks above (e.g., PyFunctionPreHook).
struct PyFunctionTensorPostAccGradHook : public PostAccumulateGradHook {
  PyFunctionTensorPostAccGradHook(PyObject* dict);
  ~PyFunctionTensorPostAccGradHook() override;
  void operator()(const Variable& tensor) override;
  // fall back to the compiled_args of PostAccumulateGradHook superclass
  PyObject* dict;
};

} // namespace autograd
} // namespace torch
