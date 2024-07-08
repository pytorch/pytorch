#pragma once

#include <torch/arg.h>
#include <torch/csrc/Export.h>
#include <torch/types.h>
#include <vector>

namespace torch {
namespace nn {

/// Options for the `LayerNorm` module.
///
/// Example:
/// ```
/// LayerNorm model(LayerNormOptions({2,
/// 2}).elementwise_affine(false).eps(2e-5));
/// ```
struct TORCH_API LayerNormOptions {
  /* implicit */ LayerNormOptions(std::vector<int64_t> normalized_shape);
  /// input shape from an expected input.
  TORCH_ARG(std::vector<int64_t>, normalized_shape);
  /// a value added to the denominator for numerical stability. ``Default:
  /// 1e-5``.
  TORCH_ARG(double, eps) = 1e-5;
  /// a boolean value that when set to ``true``, this module
  /// has learnable per-element affine parameters initialized to ones (for
  /// weights) and zeros (for biases). ``Default: true``.
  TORCH_ARG(bool, elementwise_affine) = true;
};

// ============================================================================

namespace functional {

/// Options for `torch::nn::functional::layer_norm`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::layer_norm(input, F::LayerNormFuncOptions({2, 2}).eps(2e-5));
/// ```
struct TORCH_API LayerNormFuncOptions {
  /* implicit */ LayerNormFuncOptions(std::vector<int64_t> normalized_shape);
  /// input shape from an expected input.
  TORCH_ARG(std::vector<int64_t>, normalized_shape);

  TORCH_ARG(Tensor, weight) = {};

  TORCH_ARG(Tensor, bias) = {};

  /// a value added to the denominator for numerical stability. ``Default:
  /// 1e-5``.
  TORCH_ARG(double, eps) = 1e-5;
};

} // namespace functional

// ============================================================================

/// Options for the `LocalResponseNorm` module.
///
/// Example:
/// ```
/// LocalResponseNorm
/// model(LocalResponseNormOptions(2).alpha(0.0002).beta(0.85).k(2.));
/// ```
struct TORCH_API LocalResponseNormOptions {
  /* implicit */ LocalResponseNormOptions(int64_t size) : size_(size) {}
  /// amount of neighbouring channels used for normalization
  TORCH_ARG(int64_t, size);

  /// multiplicative factor. Default: 1e-4
  TORCH_ARG(double, alpha) = 1e-4;

  /// exponent. Default: 0.75
  TORCH_ARG(double, beta) = 0.75;

  /// additive factor. Default: 1
  TORCH_ARG(double, k) = 1.;
};

namespace functional {
/// Options for `torch::nn::functional::local_response_norm`.
///
/// See the documentation for `torch::nn::LocalResponseNormOptions` class to
/// learn what arguments are supported.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::local_response_norm(x, F::LocalResponseNormFuncOptions(2));
/// ```
using LocalResponseNormFuncOptions = LocalResponseNormOptions;
} // namespace functional

// ============================================================================

/// Options for the `CrossMapLRN2d` module.
///
/// Example:
/// ```
/// CrossMapLRN2d model(CrossMapLRN2dOptions(3).alpha(1e-5).beta(0.1).k(10));
/// ```
struct TORCH_API CrossMapLRN2dOptions {
  CrossMapLRN2dOptions(int64_t size);

  TORCH_ARG(int64_t, size);

  TORCH_ARG(double, alpha) = 1e-4;

  TORCH_ARG(double, beta) = 0.75;

  TORCH_ARG(int64_t, k) = 1;
};

// ============================================================================

namespace functional {

/// Options for `torch::nn::functional::normalize`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::normalize(input, F::NormalizeFuncOptions().p(1).dim(-1));
/// ```
struct TORCH_API NormalizeFuncOptions {
  /// The exponent value in the norm formulation. Default: 2.0
  TORCH_ARG(double, p) = 2.0;
  /// The dimension to reduce. Default: 1
  TORCH_ARG(int64_t, dim) = 1;
  /// Small value to avoid division by zero. Default: 1e-12
  TORCH_ARG(double, eps) = 1e-12;
  /// the output tensor. If `out` is used, this
  /// operation won't be differentiable.
  TORCH_ARG(std::optional<Tensor>, out) = std::nullopt;
};

} // namespace functional

// ============================================================================

/// Options for the `GroupNorm` module.
///
/// Example:
/// ```
/// GroupNorm model(GroupNormOptions(2, 2).eps(2e-5).affine(false));
/// ```
struct TORCH_API GroupNormOptions {
  /* implicit */ GroupNormOptions(int64_t num_groups, int64_t num_channels);

  /// number of groups to separate the channels into
  TORCH_ARG(int64_t, num_groups);
  /// number of channels expected in input
  TORCH_ARG(int64_t, num_channels);
  /// a value added to the denominator for numerical stability. Default: 1e-5
  TORCH_ARG(double, eps) = 1e-5;
  /// a boolean value that when set to ``true``, this module
  /// has learnable per-channel affine parameters initialized to ones (for
  /// weights) and zeros (for biases). Default: ``true``.
  TORCH_ARG(bool, affine) = true;
};

// ============================================================================

namespace functional {

/// Options for `torch::nn::functional::group_norm`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::group_norm(input, F::GroupNormFuncOptions(2).eps(2e-5));
/// ```
struct TORCH_API GroupNormFuncOptions {
  /* implicit */ GroupNormFuncOptions(int64_t num_groups);

  /// number of groups to separate the channels into
  TORCH_ARG(int64_t, num_groups);

  TORCH_ARG(Tensor, weight) = {};

  TORCH_ARG(Tensor, bias) = {};

  /// a value added to the denominator for numerical stability. Default: 1e-5
  TORCH_ARG(double, eps) = 1e-5;
};

} // namespace functional

} // namespace nn
} // namespace torch
