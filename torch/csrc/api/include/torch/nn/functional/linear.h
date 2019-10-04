#pragma once

#include <torch/nn/options/activation.h>
#include <torch/types.h>

namespace torch {
namespace nn{
namespace functional {

inline Tensor linear(const Tensor& input, const Tensor& weight,
                     const Tensor& bias) {
  return torch::linear(input, weight, bias);
}

} // namespace functional
} // namespace nn
} // namespace torch
