#pragma once

#include "nnpack.h"

#include "ATen/Tensor.h"

namespace torch {
namespace nnpack {

void convolutionOutput(
    at::Tensor& input,
    at::Tensor& weight,
    at::Tensor& bias,
    const std::vector<int>& padding,
    at::Tensor& output);

} // torch::nnpack
} // torch
