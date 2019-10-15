#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/types.h>
#include <vector>

namespace torch {
namespace nn {

/// Options for the `normalize` module.
struct TORCH_API NormalizeOptions {
  /// The exponent value in the norm formulation. Default: 2.0
  TORCH_ARG(double, p) = 2.0;
  /// The dimension to reduce. Default: 1
  TORCH_ARG(int64_t, dim) = 1;
  /// Small value to avoid division by zero. Default: 1e-12
  TORCH_ARG(double, eps) = 1e-12;
};

/// Options for the `LayerNorm` module.
struct TORCH_API LayerNormOptions {
  /* implicit */ LayerNormOptions(std::vector<int64_t> normalized_shape);
  /// input shape from an expected input.
  TORCH_ARG(std::vector<int64_t>, normalized_shape);
  /// a boolean value that when set to ``True``, this module
  /// has learnable per-element affine parameters initialized to ones (for weights)
  /// and zeros (for biases).
  TORCH_ARG(bool, elementwise_affine) = true;
  /// a value added to the denominator for numerical stability.
  TORCH_ARG(double, eps) = 1e-5;
};

} // namespace nn
} // namespace torch
