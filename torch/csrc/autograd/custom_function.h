#pragma once

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <ATen/core/ivalue.h>

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
// It should take an AutogradContext* as the first argument. Variables can be
// saved in the ctx using save_for_backward() and other data can be saved in the
// map ctx.save in the form of <std::string, at::IValue> pairs.
//
// backward() should take an AutogradContext* and a variable list containing as
// many Variables as there were outputs from forward as arguments. It should
//  return as many Variables as there were inputs with each of them containing
// the gradient w.r.t. its corresponding input. Variables saved in forward can
// be accessed with ctx->get_saved_variables() and other saved data can be
// accessed from ctx->save.
//
// For example:
// class MyFunction : public CFunction<MyFunction> {
//   public:
//   static variable_list forward(AutogradContext *ctx, int n, Variable var) {
//      // Save data for backward in context
//      ctx->save["n"] = n;
//      return std::vector<Variable>({var});
//   }
//
//   static variable_list backward(AutogradContext *ctx, variable_list grad_output) {
//      // Use data saved in forward
//      auto n = ctx->save["n"].toInt();
//      return std::vector<Variable>({grad_output[0]*n});
//   }
// };
//
// To use MyFunction
// Variable x;
// MyFunction::apply(6, x);

template <class T>
struct CFunction {
  template<typename... Args>
  static variable_list apply(Args&&... args);
};


// Context to save information during forward that can be accessed in backward
struct AutogradContext {
  // Can be used to save non-variable data for backward()
  std::unordered_map<std::string, at::IValue> save;

  // Saves the list of variables for a future call to backward(). This
  // should be called at most once from inside of forward().
  void save_for_backward(const variable_list &to_save);
  // Marks variables in the list as modified in an in-place opertaion. This
  // should be called at most once from inside of forward() and all arguments
  // should be inputs.
  void mark_dirty(const variable_list &inputs);
  // Marks outputs in the list as not requiring gradients. This should be called
  // at most once from inside of forward() and all arguments should be outputs.
  void mark_non_differentiable(const variable_list &outputs);
  void clear_saved();

  // Get the list of variables that were saved in forward using
  // save_for_backward(). Before returning them to the user, a check is made to
  // ensure that they were not modified by any in-place operations.
  variable_list get_saved_variables() const;
  const std::unordered_set<at::TensorImpl*>& get_dirty() const;
  const std::unordered_set<at::TensorImpl*>& get_non_differentiable() const;

private:
  std::unordered_set<at::TensorImpl*> non_differentiable;
  std::unordered_set<at::TensorImpl*> dirty_inputs;
  std::vector<torch::autograd::SavedVariable> saved_variables;

  std::shared_ptr<Function> grad_fn;
};

template <class T>
struct CustomFunc : public Function {

  variable_list apply(variable_list&& inputs) override;
  AutogradContext ctx;

  void release_variables() override;
};
}} // namespace torch::autograd
