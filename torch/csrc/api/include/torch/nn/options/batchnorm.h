#pragma once

#include <torch/arg.h>
#include <torch/csrc/Export.h>
#include <torch/types.h>

namespace torch::nn {

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
  TORCH_ARG(std::optional<double>, momentum) = 0.1;

  /// Whether to learn a scale and bias that are applied in an affine
  /// transformation on the input.
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(bool, affine) = true;

  /// Whether to store and update batch statistics (mean and variance) in the
  /// module.
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(bool, track_running_stats) = true;
};

/// Options for the `BatchNorm1d` module.
///
/// Example:
/// ```
/// BatchNorm1d
/// model(BatchNorm1dOptions(4).eps(0.5).momentum(0.1).affine(false).track_running_stats(true));
/// ```
using BatchNorm1dOptions = BatchNormOptions;

/// Options for the `BatchNorm2d` module.
///
/// Example:
/// ```
/// BatchNorm2d
/// model(BatchNorm2dOptions(4).eps(0.5).momentum(0.1).affine(false).track_running_stats(true));
/// ```
using BatchNorm2dOptions = BatchNormOptions;

/// Options for the `BatchNorm3d` module.
///
/// Example:
/// ```
/// BatchNorm3d
/// model(BatchNorm3dOptions(4).eps(0.5).momentum(0.1).affine(false).track_running_stats(true));
/// ```
using BatchNorm3dOptions = BatchNormOptions;

// ============================================================================

namespace functional {

/// Options for `torch::nn::functional::batch_norm`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::batch_norm(input, mean, variance,
/// F::BatchNormFuncOptions().weight(weight).bias(bias).momentum(0.1).eps(1e-05).training(false));
/// ```
struct TORCH_API BatchNormFuncOptions {
  TORCH_ARG(Tensor, weight) = Tensor();

  TORCH_ARG(Tensor, bias) = Tensor();

  TORCH_ARG(bool, training) = false;

  /// A momentum multiplier for the mean and variance.
  /// Changing this parameter after construction __is effective__.
  TORCH_ARG(double, momentum) = 0.1;

  /// The epsilon value added for numerical stability.
  /// Changing this parameter after construction __is effective__.
  TORCH_ARG(double, eps) = 1e-5;
};

} // namespace functional

} // namespace torch::nn
