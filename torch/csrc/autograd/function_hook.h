#pragma once

#include <vector>

// A hook that's called on gradients

namespace torch { namespace autograd {

struct Variable;
using variable_list = std::vector<Variable>;

struct FunctionPreHook {
  virtual ~FunctionPreHook() {}
  virtual variable_list operator()(const variable_list& grads) = 0;
};

struct FunctionPostHook {
  virtual ~FunctionPostHook() {}
  virtual variable_list operator()(const variable_list& grad_input, const variable_list& grad_output) = 0;
};

}} // namespace torch::autograd
