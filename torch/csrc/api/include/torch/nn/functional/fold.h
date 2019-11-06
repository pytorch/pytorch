#pragma once

#include <torch/nn/options/fold.h>

namespace torch {
namespace nn {
namespace functional {

inline Tensor fold(const Tensor& input, const FoldFuncOptions& options) {
  if (input.dim() == 3) {
    return torch::col2im(
        input,
        options.output_size(),
        options.kernel_size(),
        options.dilation(),
        options.padding(),
        options.stride());
  } else {
    TORCH_CHECK(
        false,
        "Input Error: Only 3D input Tensors are supported "
        "(got ", input.dim(), "D)");
  }
}

inline Tensor unfold(const Tensor& input, const UnfoldFuncOptions& options) {
  if (input.dim() == 4) {
    return torch::im2col(
        input,
        options.kernel_size(),
        options.dilation(),
        options.padding(),
        options.stride());
  } else {
    TORCH_CHECK(
        false,
        "Input Error: Only 4D input Tensors are supported "
        "(got ", input.dim(), "D)");
  }
}

} // namespace functional
} // namespace nn
} // namespace torch
