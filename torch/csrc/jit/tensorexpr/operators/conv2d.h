#pragma once

#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch {
namespace jit {
namespace tensorexpr {

TORCH_API Tensor* conv2d_depthwise(
    BufHandle input,
    BufHandle weight,
    BufHandle bias,
    int stride,
    int pad,
    int groups);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
