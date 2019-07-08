#pragma once

#include <vector>
#include <torch/csrc/WindowsTorchApiMacro.h>

// A hook that's called on gradients

namespace torch { namespace autograd {

struct Variable;
using variable_list = std::vector<Variable>;

struct TORCH_API FunctionPreHook {
  virtual ~FunctionPreHook();
  virtual variable_list operator()(const variable_list& grads) = 0;
};

struct TORCH_API FunctionPostHook {
  virtual ~FunctionPostHook();
  virtual variable_list operator()(
    const variable_list& outputs /* grad_inputs */,
    const variable_list& inputs /* grad_outputs */) = 0;
};

}} // namespace torch::autograd
