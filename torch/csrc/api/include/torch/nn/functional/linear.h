#pragma once

#include <torch/nn/options/activation.h>
#include <torch/types.h>

namespace torch {
namespace nn {
namespace functional {

inline Tensor linear(const Tensor& input, const Tensor& weight,
                     const Tensor& bias = {}) {
  if (input.dim() == 2 and bias.defined()) {
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
