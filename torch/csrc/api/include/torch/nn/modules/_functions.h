#pragma once

#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/types.h>
#include <torch/nn/options/normalization.h>

namespace torch {
namespace nn {
namespace functions {

class CrossMapLRN2d : public torch::autograd::Function<CrossMapLRN2d> {
 public:
  static torch::autograd::Variable forward(
      torch::autograd::AutogradContext *ctx,
      const torch::autograd::Variable& input,
      const CrossMapLRN2dOptions& options);
  
  static torch::autograd::variable_list backward(
    torch::autograd::AutogradContext *ctx, torch::autograd::variable_list grad_output);
};

} // namespace functions
} // namespace nn
} // namespace torch
