#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/nn/options/common.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Options for the `BatchNorm` module.
struct TORCH_API BatchNormOptions {
  /* implicit */ BatchNormOptions(int64_t num_features);

  /// The number of features of the input tensor.
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(int64_t, num_features);

  /// The epsilon value added for numerical stability.
  /// Changing this parameter after construction __is effective__.
  TORCH_ARG(double, eps) = 1e-5;

  /// A momentum multiplier for the mean and variance.
  /// Changing this parameter after construction __is effective__.
  TORCH_ARG(c10::optional<double>, momentum) = 0.1;

  /// Whether to learn a scale and bias that are applied in an affine
  /// transformation on the input.
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(bool, affine) = true;

  /// Whether to store and update batch statistics (mean and variance) in the
  /// module.
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(bool, track_running_stats) = true;
};

using BatchNorm1dOptions = BatchNormOptions;
using BatchNorm2dOptions = BatchNormOptions;
using BatchNorm3dOptions = BatchNormOptions;

// ============================================================================

namespace functional {

/// Options for the `BatchNorm` functional.
struct TORCH_API BatchNormFuncOptions {
  TORCH_ARG(Tensor, weight) = Tensor();

  TORCH_ARG(Tensor, bias) = Tensor();

  TORCH_ARG(bool, training) = false;

  /// A momentum multiplier for the mean and variance.
  /// Changing this parameter after construction __is effective__.
  TORCH_ARG(c10::optional<double>, momentum) = 0.1;

  /// The epsilon value added for numerical stability.
  /// Changing this parameter after construction __is effective__.
  TORCH_ARG(double, eps) = 1e-5;
};

} // namespace functional

} // namespace nn
} // namespace torch
