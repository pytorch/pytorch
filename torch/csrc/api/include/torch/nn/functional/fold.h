#pragma once

#include <torch/nn/options/fold.h>

namespace torch {
namespace nn {
namespace functional {

namespace detail {
inline Tensor fold(const Tensor& input,
                   ExpandingArray<2> output_size,
                   ExpandingArray<2> kernel_size,
                   ExpandingArray<2> dilation = 1,
                   ExpandingArray<2> padding = 0,
                   ExpandingArray<2> stride = 1) {
  if (input.dim() == 3) {
    return torch::col2im(
        input,
        output_size,
        kernel_size,
        dilation,
        padding,
        stride);
  } else {
    TORCH_CHECK(
        false,
        "Input Error: Only 3D input Tensors are supported "
        "(got ", input.dim(), "D)");
  }
}
} // namespace detail

inline Tensor fold(const Tensor& input, FoldFuncOptions options) {
  return detail::fold(
    input,
    options.output_size(),
    options.kernel_size(),
    options.dilation(),
    options.padding(),
    options.stride());
}

// ============================================================================

namespace detail {
inline Tensor unfold(const Tensor& input,
                     ExpandingArray<2> kernel_size,
                     ExpandingArray<2> dilation = 1,
                     ExpandingArray<2> padding = 0,
                     ExpandingArray<2> stride = 1) {
  if (input.dim() == 4) {
    return torch::im2col(
        input,
        kernel_size,
        dilation,
        padding,
        stride);
  } else {
    TORCH_CHECK(
        false,
        "Input Error: Only 4D input Tensors are supported "
        "(got ", input.dim(), "D)");
  }
}
} // namespace detail

inline Tensor unfold(const Tensor& input, UnfoldFuncOptions options) {
  return detail::unfold(input, options.kernel_size(), options.dilation(), options.padding(), options.stride());
}

} // namespace functional
} // namespace nn
} // namespace torch
