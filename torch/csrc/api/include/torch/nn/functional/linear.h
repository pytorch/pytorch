#pragma once

#include <torch/types.h>

namespace torch {
namespace nn {
namespace functional {

inline Tensor bilinear(const Tensor& input1, const Tensor& input2, const Tensor& weight, const Tensor& bias=Tensor()) {
    return torch::bilinear(input1, input2, weight, bias);
}

// ============================================================================

inline Tensor linear(const Tensor& input, const Tensor& weight,
                     const Tensor& bias = {}, const optional<int64_t> axis = {}) {
  Tensor new_input = input;
  if (axis.has_value()) {
    int _axis = axis.value();
    auto sizes = new_input.sizes().vec();
    sizes.resize(_axis + 1);
    sizes[_axis] = weight.sizes().vec()[1];
    new_input = new_input.view(sizes);
  }

  if (new_input.dim() == 2 && bias.defined()) {
    // fused op is marginally faster
    return torch::addmm(bias, new_input, weight.t());
  } else {
    auto output = new_input.matmul(weight.t());
    if (bias.defined()) {
        output += bias;
    }
    return output;
  }
}

} // namespace functional
} // namespace nn
} // namespace torch
