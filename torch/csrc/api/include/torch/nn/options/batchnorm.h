#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Options for the `BatchNorm` module.
struct TORCH_API BatchNormOptions {
  /* implicit */ BatchNormOptions(int64_t features);
  /// The number of features of the input tensor.
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(int64_t, features);
  /// Whether to learn a scale and bias that are applied in an affine
  /// transformation on the input.
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(bool, affine) = true;
  /// Whether to store and update batch statistics (mean and variance) in the
  /// module. If `false`, you should call `pure_forward` and supply those batch
  /// statistics yourself.
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(bool, stateful) = true;
  /// The epsilon value added for numerical stability.
  /// Changing this parameter after construction __is effective__.
  TORCH_ARG(double, eps) = 1e-5;
  /// A momentum multiplier for the mean and variance.
  /// Changing this parameter after construction __is effective__.
  TORCH_ARG(double, momentum) = 0.1;
};

} // namespace nn
} // namespace torch
