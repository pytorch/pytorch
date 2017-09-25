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

void convolutionUpdateGradInput(
    at::Tensor& input,
    at::Tensor& weight,
    const std::vector<int>& padding,
    at::Tensor& gradOutput,
    at::Tensor& gradInput);

void convolutionUpdateGradWeight(
    at::Tensor& input,
    at::Tensor& gradWeight,
    const std::vector<int>& padding,
    at::Tensor& gradOutput);

} // torch::nnpack
} // torch
