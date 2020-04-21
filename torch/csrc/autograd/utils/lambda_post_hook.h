#pragma once

#include <torch/csrc/autograd/function_hook.h>

namespace torch {
namespace autograd {
namespace utils {

// Turns lambda into a torch::autograd::FunctionPostHook.
class LambdaPostHook : public torch::autograd::FunctionPostHook {
  using variable_list = std::vector<torch::autograd::Variable>;

 public:
  /* implicit */ LambdaPostHook(std::function<void(const variable_list&)> fn)
      : fn_(std::move(fn)) {}

  variable_list operator()(
      const variable_list& outputs,
      const variable_list& inputs) override {
    fn_(inputs);
    return outputs;
  }

 protected:
  std::function<void(const variable_list&)> fn_;
};

} // namespace utils
} // namespace autograd
} // namespace torch
