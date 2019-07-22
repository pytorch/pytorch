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

// To use custom autograd operations implement a CFunction subclass with
// static backward and forward functions
//
// forward() can take as many arguments as you want and should return a
// variable list. Use of any direct Variable arguments will be registered in
// the graph but no vectors/sets or any other data structures will be traversed.
//
// backward() will be given a variable list containing as many Variables as
// there were outputs from forward. It should return as many Variables as there
// were inputs with each of them containing the gradient w.r.t. its
// corresponding input
//
// For example:
// class MyFunction : public CFunction<MyFunction> {
//   public:
//   static variable_list forward(AutogradContext *ctx, int n, Variable var);
//
//   static variable_list backward(AutogradContext *ctx, variable_list grad_output);
// };
// To use MyFunction
// Variable x;
// MyFunction::apply(6, x);

template <class T>
struct CFunction {
  template<typename... Args>
  static variable_list apply(Args&&... args);
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
