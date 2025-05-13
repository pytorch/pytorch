#pragma once

#include <torch/csrc/autograd/function_hook.h>
#include <torch/csrc/dynamo/compiled_autograd.h>

namespace torch::autograd::utils {

// Turns lambda into a torch::autograd::FunctionPostHook.
class LambdaPostHook : public torch::autograd::FunctionPostHook {
  using variable_list = std::vector<torch::autograd::Variable>;
  using fn_type =
      std::function<variable_list(const variable_list&, const variable_list&)>;
  using compiled_args_fn_type = std::function<void(CompiledNodeArgs&)>;
  using compiled_apply_fn_type =
      std::function<void(Variable&, SwapSavedVariables&)>;

 public:
  // The lambda function takes as arguments the outputs and inputs of the
  // autograd function and can modify the outputs of the autograd function by
  // returning a new output if needed.
  /* implicit */ LambdaPostHook(fn_type fn) : fn_(std::move(fn)) {}

  LambdaPostHook(
      fn_type fn,
      compiled_args_fn_type compiled_args_fn,
      compiled_apply_fn_type compiled_apply_fn)
      : fn_(std::move(fn)),
        compiled_args_fn_(std::move(compiled_args_fn)),
        compiled_apply_fn_(std::move(compiled_apply_fn)) {}

  variable_list operator()(
      const variable_list& outputs,
      const variable_list& inputs) override {
    return fn_(outputs, inputs);
  }

  void compiled_args(CompiledNodeArgs& args) const override {
    if (compiled_args_fn_ != nullptr) {
      return compiled_args_fn_(args);
    }
    return FunctionPostHook::compiled_args(args);
  }

  void apply_with_saved(Variable& inputs, SwapSavedVariables& saved)
      const override {
    if (compiled_apply_fn_ != nullptr) {
      return compiled_apply_fn_(inputs, saved);
    }
    return FunctionPostHook::apply_with_saved(inputs, saved);
  }

 protected:
  std::function<variable_list(const variable_list&, const variable_list&)> fn_;
  compiled_args_fn_type compiled_args_fn_{};
  compiled_apply_fn_type compiled_apply_fn_{};
};

} // namespace torch::autograd::utils
