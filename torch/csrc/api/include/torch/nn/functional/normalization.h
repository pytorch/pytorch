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

} // namespace functional
} // namespace nn
} // namespace torch
