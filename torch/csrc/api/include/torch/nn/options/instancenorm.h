#pragma once

#include <torch/arg.h>
#include <torch/csrc/Export.h>
#include <torch/nn/options/batchnorm.h>
#include <torch/types.h>

namespace torch::nn {

/// Options for the `InstanceNorm` module.
struct TORCH_API InstanceNormOptions {
  /* implicit */ InstanceNormOptions(int64_t num_features);

  /// The number of features of the input tensor.
  TORCH_ARG(int64_t, num_features);

  /// The epsilon value added for numerical stability.
  TORCH_ARG(double, eps) = 1e-5;

  /// A momentum multiplier for the mean and variance.
  TORCH_ARG(double, momentum) = 0.1;

  /// Whether to learn a scale and bias that are applied in an affine
  /// transformation on the input.
  TORCH_ARG(bool, affine) = false;

  /// Whether to store and update batch statistics (mean and variance) in the
  /// module.
  TORCH_ARG(bool, track_running_stats) = false;
};

/// Options for the `InstanceNorm1d` module.
///
/// Example:
/// ```
/// InstanceNorm1d
/// model(InstanceNorm1dOptions(4).eps(0.5).momentum(0.1).affine(false).track_running_stats(true));
/// ```
using InstanceNorm1dOptions = InstanceNormOptions;

/// Options for the `InstanceNorm2d` module.
///
/// Example:
/// ```
/// InstanceNorm2d
/// model(InstanceNorm2dOptions(4).eps(0.5).momentum(0.1).affine(false).track_running_stats(true));
/// ```
using InstanceNorm2dOptions = InstanceNormOptions;

/// Options for the `InstanceNorm3d` module.
///
/// Example:
/// ```
/// InstanceNorm3d
/// model(InstanceNorm3dOptions(4).eps(0.5).momentum(0.1).affine(false).track_running_stats(true));
/// ```
using InstanceNorm3dOptions = InstanceNormOptions;

namespace functional {

/// Options for `torch::nn::functional::instance_norm`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::instance_norm(input,
/// F::InstanceNormFuncOptions().running_mean(mean).running_var(variance).weight(weight).bias(bias).momentum(0.1).eps(1e-5));
/// ```
struct TORCH_API InstanceNormFuncOptions {
  TORCH_ARG(Tensor, running_mean) = Tensor();

  TORCH_ARG(Tensor, running_var) = Tensor();

  TORCH_ARG(Tensor, weight) = Tensor();

  TORCH_ARG(Tensor, bias) = Tensor();

  TORCH_ARG(bool, use_input_stats) = true;

  TORCH_ARG(double, momentum) = 0.1;

  TORCH_ARG(double, eps) = 1e-5;
};

} // namespace functional

} // namespace torch::nn
