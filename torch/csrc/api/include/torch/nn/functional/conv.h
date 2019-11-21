#pragma once

#include <torch/nn/options/conv.h>
#include <torch/types.h>

namespace torch {
namespace nn {
namespace functional {

namespace detail {
inline Tensor conv1d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    ExpandingArray<1> stride,
    ExpandingArray<1> padding,
    ExpandingArray<1> dilation,
    int64_t groups) {
  return torch::conv1d(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    groups);
}
} // namespace detail

inline Tensor conv1d(
    const Tensor& input,
    const Tensor& weight,
    const Conv1dFuncOptions& options = {}) {
  return detail::conv1d(
    input,
    weight,
    options.bias(),
    options.stride(),
    options.padding(),
    options.dilation(),
    options.groups());
}

namespace detail {
inline Tensor conv2d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    ExpandingArray<2> stride,
    ExpandingArray<2> padding,
    ExpandingArray<2> dilation,
    int64_t groups) {
  return torch::conv2d(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    groups);
}
} // namespace detail

inline Tensor conv2d(
    const Tensor& input,
    const Tensor& weight,
    const Conv2dFuncOptions& options = {}) {
  return detail::conv2d(
    input,
    weight,
    options.bias(),
    options.stride(),
    options.padding(),
    options.dilation(),
    options.groups());
}

namespace detail {
inline Tensor conv3d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    ExpandingArray<3> stride,
    ExpandingArray<3> padding,
    ExpandingArray<3> dilation,
    int64_t groups) {
  return torch::conv3d(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    groups);
}
} // namespace detail

inline Tensor conv3d(
    const Tensor& input,
    const Tensor& weight,
    const Conv3dFuncOptions& options = {}) {
  return detail::conv3d(
    input,
    weight,
    options.bias(),
    options.stride(),
    options.padding(),
    options.dilation(),
    options.groups());
}

} // namespace functional
} // namespace nn
} // namespace torch
