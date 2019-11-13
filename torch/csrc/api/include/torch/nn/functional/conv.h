#pragma once

#include <torch/types.h>

namespace torch {
namespace nn {
namespace functional {

inline Tensor conv_transpose1d(const Tensor& input, const Tensor& weight,
                               const Tensor& bias, IntArrayRef stride,
                               IntArrayRef padding, IntArrayRef output_padding,
                               int64_t groups, IntArrayRef dilation) {
  return torch::conv_transpose1d(
      input, weight, bias, stride, padding, output_padding, groups, dilation);
}

} // namespace functional
} // namespace nn
} // namespace torch
