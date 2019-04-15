#pragma once

#include <vector>

// A hook that's called on gradients

namespace torch { namespace autograd {

struct Variable;
using variable_list = std::vector<Variable>;

struct FunctionPreHook {
  virtual ~FunctionPreHook() = default;
  virtual variable_list operator()(const variable_list& grads) = 0;
};

struct FunctionPostHook {
  virtual ~FunctionPostHook() = default;
  virtual variable_list operator()(
    const variable_list& outputs /* grad_inputs */,
    const variable_list& inputs /* grad_outputs */) = 0;
};

}} // namespace torch::autograd
