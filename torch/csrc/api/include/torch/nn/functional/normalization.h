#pragma once

#include <torch/nn/functional/padding.h>
#include <torch/nn/functional/pooling.h>
#include <torch/nn/options/normalization.h>
#include <torch/types.h>

namespace torch::nn::functional {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor normalize(
    const Tensor& input,
    double p,
    int64_t dim,
    double eps,
    std::optional<Tensor> out) {
  if (out == std::nullopt) {
    auto denom = input.norm(p, dim, true).clamp_min(eps).expand_as(input);
    return input / denom;
  } else {
    auto denom = input.norm(p, dim, true).clamp_min(eps).expand_as(input);
    return torch::div_out(*out, input, denom);
  }
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.normalize
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::NormalizeFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::normalize(input, F::NormalizeFuncOptions().p(1).dim(-1));
/// ```
inline Tensor normalize(
    const Tensor& input,
    NormalizeFuncOptions options = {}) {
  return detail::normalize(
      input, options.p(), options.dim(), options.eps(), options.out());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor layer_norm(
    const Tensor& input,
    const std::vector<int64_t>& normalized_shape,
    const Tensor& weight,
    const Tensor& bias,
    double eps) {
  return torch::layer_norm(input, normalized_shape, weight, bias, eps);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.layer_norm
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::LayerNormFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::layer_norm(input, F::LayerNormFuncOptions({2, 2}).eps(2e-5));
/// ```
inline Tensor layer_norm(
    const Tensor& input,
    const LayerNormFuncOptions& options) {
  return detail::layer_norm(
      input,
      options.normalized_shape(),
      options.weight(),
      options.bias(),
      options.eps());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor local_response_norm(
    const Tensor& input,
    int64_t size,
    double alpha,
    double beta,
    double k) {
  auto dim = input.dim();
  TORCH_CHECK(
      dim >= 3,
      "Expected 3D or higher dimensionality input (got ",
      dim,
      " dimensions)");
  auto div = input.mul(input).unsqueeze(1);
  if (dim == 3) {
    div = detail::pad(
        div,
        /*pad=*/{0, 0, size / 2, (size - 1) / 2},
        /*mode=*/torch::kConstant,
        /*value=*/0);
    div = detail::avg_pool2d(
              div,
              /*kernel_size=*/{size, 1},
              /*stride=*/1,
              /*padding=*/0,
              /*ceil_mode=*/false,
              /*count_include_pad=*/true,
              /*divisor_override=*/std::nullopt)
              .squeeze(1);
  } else {
    auto sizes = input.sizes();
    div = div.view({sizes[0], 1, sizes[1], sizes[2], -1});
    div = detail::pad(
        div,
        /*pad=*/{0, 0, 0, 0, size / 2, (size - 1) / 2},
        /*mode=*/torch::kConstant,
        /*value=*/0);
    div = detail::avg_pool3d(
              div,
              /*kernel_size=*/{size, 1, 1},
              /*stride=*/1,
              /*padding=*/0,
              /*ceil_mode=*/false,
              /*count_include_pad=*/true,
              /*divisor_override=*/std::nullopt)
              .squeeze(1);
    div = div.view(sizes);
  }
  div = div.mul(alpha).add(k).pow(beta);
  return input / div;
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.local_response_norm
/// about the exact behavior of this functional.
///
/// See the documentation for
/// `torch::nn::functional::LocalResponseNormFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::local_response_norm(x, F::LocalResponseNormFuncOptions(2));
/// ```
inline Tensor local_response_norm(
    const Tensor& input,
    const LocalResponseNormFuncOptions& options) {
  return detail::local_response_norm(
      input, options.size(), options.alpha(), options.beta(), options.k());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor group_norm(
    const Tensor& input,
    int64_t num_groups,
    const Tensor& weight,
    const Tensor& bias,
    double eps) {
  return torch::group_norm(
      input,
      num_groups,
      weight,
      bias,
      eps,
      at::globalContext().userEnabledCuDNN());
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.group_norm
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::GroupNormFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::group_norm(input, F::GroupNormFuncOptions(2).eps(2e-5));
/// ```
inline Tensor group_norm(
    const Tensor& input,
    const GroupNormFuncOptions& options) {
  return detail::group_norm(
      input,
      options.num_groups(),
      options.weight(),
      options.bias(),
      options.eps());
}

} // namespace torch::nn::functional
