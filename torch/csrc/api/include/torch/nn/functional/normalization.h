#pragma once

#include <torch/nn/options/normalization.h>
#include <torch/nn/functional/padding.h>
#include <torch/nn/functional/pooling.h>
#include <torch/types.h>

namespace torch {
namespace nn {
namespace functional {

inline Tensor normalize(
    const Tensor& input,
    const NormalizeFuncOptions& options = {},
    c10::optional<Tensor> out = c10::nullopt) {
    if (out == c10::nullopt) {
      auto denom = input.norm(options.p(), options.dim(), true).clamp_min(options.eps()).expand_as(input);
      return input / denom;
    } else {
      auto denom = input.norm(options.p(), options.dim(), true).clamp_min(options.eps()).expand_as(input);
      return torch::div_out(*out, input, denom);
    }
}

inline Tensor layer_norm(const Tensor& input,
    const LayerNormFuncOptions& options,
    const Tensor& weight = Tensor(),
    const Tensor& bias = Tensor()) {

    return torch::layer_norm(input, options.normalized_shape(), weight, bias, options.eps());
}

inline Tensor local_response_norm(
    const Tensor& input,
    const LocalResponseNormFuncOptions& options) {
    auto dim = input.dim();
    TORCH_CHECK(dim >=3, "Expected 3D or higher dimensionality input (got ", dim, " dimensions)");
    auto div = input.mul(input).unsqueeze(1);
    if (dim == 3) {
      div = pad(div, PadFuncOptions({0, 0, options.size() / 2, (options.size() - 1) / 2}));
      div = avg_pool2d(div, AvgPool2dFuncOptions({options.size(), 1}).stride(1)).squeeze(1);
    } else {
      auto sizes = input.sizes();
      div = div.view({sizes[0], 1, sizes[1], sizes[2], -1});
      div = pad(div, PadFuncOptions({0, 0, 0, 0, options.size() / 2, (options.size() - 1) / 2}));
      div = avg_pool3d(div, AvgPool3dFuncOptions({options.size(), 1, 1}).stride(1)).squeeze(1);
      div = div.view(sizes);
    }
    div = div.mul(options.alpha()).add(options.k()).pow(options.beta());
    return input / div;
}

} // namespace functional
} // namespace nn
} // namespace torch
