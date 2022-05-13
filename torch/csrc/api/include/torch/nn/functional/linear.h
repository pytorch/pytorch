#pragma once

#include <torch/types.h>

namespace torch {
namespace nn {
namespace functional {

inline Tensor bias(const Tensor& input, const Tensor& bias){
  return torch::bias(input, bias);
}

inline Tensor bilinear(const Tensor& input1, const Tensor& input2, const Tensor& weight, const Tensor& bias=Tensor()) {
    return torch::bilinear(input1, input2, weight, bias);
}

// ============================================================================

inline Tensor linear(const Tensor& input, const Tensor& weight,
                     const Tensor& bias = {}) {
  if (input.dim() == 2 && bias.defined()) {
    // fused op is marginally faster
    return torch::addmm(bias, input, weight.t());
  } else {
    auto output = input.matmul(weight.t());
    if (bias.defined()) {
        output += bias;
    }
    return output;
  }
}

} // namespace functional
} // namespace nn
} // namespace torch
