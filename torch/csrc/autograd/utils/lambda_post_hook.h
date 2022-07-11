#pragma once

#include <torch/csrc/autograd/function_hook.h>

namespace torch {
namespace autograd {
namespace utils {

// Turns lambda into a torch::autograd::FunctionPostHook.
class LambdaPostHook : public torch::autograd::FunctionPostHook {
  using variable_list = std::vector<torch::autograd::Variable>;

 public:
  // The lambda function takes as arguments the outputs and inputs of the
  // autograd function and can modify the outputs of the autograd function by
  // returning a new output if needed.
  /* implicit */ LambdaPostHook(
      std::function<variable_list(const variable_list&, const variable_list&)>
          fn)
      : fn_(std::move(fn)) {}

  variable_list operator()(
      const variable_list& outputs,
      const variable_list& inputs) override {
    return fn_(outputs, inputs);
  }

 protected:
  std::function<variable_list(const variable_list&, const variable_list&)> fn_;
};

} // namespace utils
} // namespace autograd
} // namespace torch
