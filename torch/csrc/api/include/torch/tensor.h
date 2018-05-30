#pragma once

#include <ATen/ATen.h>

#include <torch/csrc/autograd/variable.h>

// This is all temporary until I create a nice torch::Tensor API.

namespace torch {
using Variable = autograd::Variable;

inline Variable Var(at::Tensor data, bool requires_grad = true) {
  return autograd::make_variable(data, requires_grad);
}
} // namespace torch
