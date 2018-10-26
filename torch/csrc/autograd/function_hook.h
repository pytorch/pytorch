#pragma once

#include <vector>

#include "torch/csrc/WindowsTorchApiMacro.h"

// A hook that's called on gradients

namespace torch { namespace autograd {

struct Variable;
using variable_list = std::vector<Variable>;

struct FunctionPreHook {
  virtual TORCH_API ~FunctionPreHook();
  virtual variable_list operator()(const variable_list& grads) = 0;
};

struct FunctionPostHook {
  virtual TORCH_API ~FunctionPostHook();
  virtual variable_list operator()(const variable_list& grad_input, const variable_list& grad_output) = 0;
};

}} // namespace torch::autograd
