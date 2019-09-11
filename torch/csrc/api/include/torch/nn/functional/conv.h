#pragma once

#include <torch/nn/options/conv.h>
#include <torch/types.h>

namespace torch {
namespace nn{
namespace functional {

inline Tensor conv1d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias = {},
    const Conv1dOptions& options = {}) {
  return torch::conv1d(
    input,
    weight,
    bias,
    options.stride(),
    options.padding(),
    options.dilation(),
    options.groups());
}

inline Tensor conv2d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias = {},
    const Conv2dOptions& options = {}) {
  return torch::conv2d(
    input,
    weight,
    bias,
    options.stride(),
    options.padding(),
    options.dilation(),
    options.groups());
}

inline Tensor conv3d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias = {},
    const Conv3dOptions& options = {}) {
  return torch::conv3d(
    input,
    weight,
    bias,
    options.stride(),
    options.padding(),
    options.dilation(),
    options.groups());
}

} // namespace functional
} // namespace nn
} // namespace torch
