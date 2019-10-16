#pragma once

#include <torch/types.h>

namespace torch {
namespace nn{
namespace functional {

inline Tensor bilinear(const Tensor& input1, const Tensor& input2, const Tensor& weight, const Tensor& bias=Tensor()) {
    return torch::bilinear(input1, input2, weight, bias);
}

} // namespace functional
} // namespace nn
} // namespace torch
