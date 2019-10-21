#pragma once

#include <torch/nn/options/normalization.h>

namespace torch {
namespace nn {
namespace functional {

inline Tensor normalize(
    const Tensor& input,
    const NormalizeOptions& options = {},
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
    const LayerNormOptions& options,
    const Tensor& weight = Tensor(),
    const Tensor& bias = Tensor()) {

    return torch::layer_norm(input, options.normalized_shape(), weight, bias, options.eps());
}

} // namespace functional
} // namespace nn
} // namespace torch
