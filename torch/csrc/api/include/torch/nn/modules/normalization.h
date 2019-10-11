#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Options for the `BatchNorm` module.
struct TORCH_API LayerNormOptions {
  /* implicit */ LayerNormOptions(int64_t normalized_shape);
  /// The number of features of the input tensor.
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(int64_t, normalized_shape);
  /// a boolean value that when set to ``True``, this module
  /// has learnable per-element affine parameters initialized to ones (for weights)
  /// and zeros (for biases).
  TORCH_ARG(bool, elementwise_affine) = true;
  /// The epsilon value added for numerical stability.
  /// Changing this parameter after construction __is effective__.
  TORCH_ARG(double, eps) = 1e-5;
};

} // namespace nn
} // namespace torch
