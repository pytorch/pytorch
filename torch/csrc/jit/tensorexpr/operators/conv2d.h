#pragma once

#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch {
namespace jit {
namespace tensorexpr {

// An API to compute 2D depthwise convolutions with bias and static shapes
// on all parameters.
TORCH_API Tensor* conv2d_depthwise(
    BufHandle input,
    BufHandle weight,
    BufHandle bias,
    int stride,
    int pad,
    int groups);

// An API to compute 2D depthwise convolutions without bias and static shapes
// on all parameters.
TORCH_API Tensor* conv2d_depthwise(
    BufHandle input,
    BufHandle weight,
    int stride,
    int pad,
    int groups);

// APIs to compute 2D depthwise convolutions with dynamic shape on input
// and static shapes on all other parameters.
TORCH_API Tensor* conv2d_depthwise(
    BufHandle input,
    BufHandle weight,
    BufHandle bias,
    ExprHandle N,
    ExprHandle C,
    ExprHandle H,
    ExprHandle W,
    int stride,
    int pad,
    int groups);

TORCH_API Tensor* conv2d_depthwise(
    BufHandle input,
    BufHandle weight,
    ExprHandle N,
    ExprHandle C,
    ExprHandle H,
    ExprHandle W,
    int stride,
    int pad,
    int groups);

// APIs to compute 2D depthwise convolutions with dynamic shapes on input,
// the first two dims on weight, and groups, while all other parameters are
// statically known.
TORCH_API Tensor* conv2d_depthwise(
    BufHandle input,
    BufHandle weight,
    BufHandle bias,
    ExprHandle N,
    ExprHandle C,
    ExprHandle H,
    ExprHandle W,
    ExprHandle K,
    ExprHandle CperG,
    int stride,
    int pad,
    ExprHandle groups);

TORCH_API Tensor* conv2d_depthwise(
    BufHandle input,
    BufHandle weight,
    ExprHandle N,
    ExprHandle C,
    ExprHandle H,
    ExprHandle W,
    ExprHandle K,
    ExprHandle CperG,
    int stride,
    int pad,
    ExprHandle groups);

// APIs to compute 2D depthwise convolutions with dynamic shapes on all
// parameters.
TORCH_API Tensor* conv2d_depthwise(
    BufHandle input,
    BufHandle weight,
    BufHandle bias,
    ExprHandle N,
    ExprHandle C,
    ExprHandle H,
    ExprHandle W,
    ExprHandle K,
    ExprHandle CperG,
    ExprHandle R,
    ExprHandle S,
    ExprHandle stride,
    ExprHandle pad,
    ExprHandle groups);

TORCH_API Tensor* conv2d_depthwise(
    BufHandle input,
    BufHandle weight,
    ExprHandle N,
    ExprHandle C,
    ExprHandle H,
    ExprHandle W,
    ExprHandle K,
    ExprHandle CperG,
    ExprHandle R,
    ExprHandle S,
    ExprHandle stride,
    ExprHandle pad,
    ExprHandle groups);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
