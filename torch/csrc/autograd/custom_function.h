#pragma once

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>

namespace torch { namespace autograd {

TORCH_API variable_list _wrap_outputs(
  const variable_list &input_vars,
  const std::unordered_set<at::TensorImpl*> &non_differentiable,
  const std::unordered_set<at::TensorImpl*> &dirty_inputs,
  const at::ArrayRef<Variable> raw_outputs,
  const std::shared_ptr<Function> &cdata);

// To use custom autograd operations implement a CFunction subclass.
// class MyFunction : public CFunction {
//   static variable_list forward(AutogradContext *ctx, variable_list inputs);
//
//   static variable_list backward(AutogradContext *ctx, variable_list grad_output);
// };
// To use MyFunction
// MyFunction::apply(inputs)

template <class T>
struct CFunction {
  static void apply(variable_list&& inputs);
};

struct AutogradContext {
  std::vector<torch::autograd::SavedVariable> saved_variables;

  std::unordered_set<at::TensorImpl*> non_differentiable;
  std::unordered_set<at::TensorImpl*> dirty_inputs;
};

template <class T>
struct CustomFunc : public Function {

  variable_list apply(variable_list&& inputs) override;
  AutogradContext ctx;

  void release_variables() override;
};

}} // namespace torch::autograd
