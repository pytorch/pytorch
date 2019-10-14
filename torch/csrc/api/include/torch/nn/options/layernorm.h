#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Options for the `LayerNorm` module.
struct TORCH_API LayerNormOptions {
  /* implicit */ LayerNormOptions(torch::IntArrayRef normalized_shape);
  /// input shape from an expected input.
  TORCH_ARG(torch::IntArrayRef, normalized_shape);
  /// a boolean value that when set to ``True``, this module
  /// has learnable per-element affine parameters initialized to ones (for weights)
  /// and zeros (for biases).
  TORCH_ARG(bool, elementwise_affine) = true;
  /// a value added to the denominator for numerical stability.
  TORCH_ARG(double, eps) = 1e-5;
};
//std::vector<int64_t>
} // namespace nn
} // namespace torch
