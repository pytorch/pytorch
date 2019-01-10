#pragma once

#include <torch/types.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace nn {
namespace init {

/// Fills the given `tensor` with the provided `value` in-place, and returns it.
/// No gradient will be recorded for this operation.
TORCH_API Tensor constant_(Tensor tensor, Scalar value);

/// Fills the given `tensor` with the Dirac delta function in-place, and returns
/// it. No gradient will be recorded for this operation.
TORCH_API Tensor dirac_(Tensor tensor);

/// Fills the given 2-dimensional `matrix` with an identity matrix.
/// No gradient will be recorded for this operation.
TORCH_API Tensor eye_(Tensor matrix);

/// Fills the given 2-dimensional `matrix` with values drawn from a normal
/// distribution parameterized by `mean` and `std`.
/// No gradient will be recorded for this operation.
TORCH_API Tensor normal_(Tensor tensor, double mean = 0, double std = 1);

/// Fills the given `tensor` with ones.
/// No gradient will be recorded for this operation.
TORCH_API Tensor ones_(Tensor tensor);

/// Fills the input `Tensor` with a (semi) orthogonal matrix, as described in
/// "Exact solutions to the nonlinear dynamics of learning in deep linear neural
/// networks" - Saxe, A. et al. (2013). The input tensor must have at least 2
/// dimensions, and for tensors with more than 2 dimensions the trailing
/// dimensions are flattened.
/// No gradient will be recorded for this operation.
TORCH_API Tensor orthogonal_(Tensor tensor, double gain = 1.0);

/// Fills the 2D input `Tensor` as a sparse matrix, where the
/// non-zero elements will be drawn from a centered normal distribution
/// with the given standard deviation `std`, as described in "Deep learning via
/// Hessian-free optimization" - Martens, J. (2010). The `sparsity` is a real
/// value between 0 and 1 that controls the fraction of elements in each column
/// to be set to zero.
/// No gradient will be recorded for this operation.
TORCH_API Tensor sparse_(Tensor tensor, double sparsity, double std = 0.01);

/// Fills the given 2-dimensional `matrix` with values drawn from a uniform
/// distribution parameterized by `low` and `high`.
/// No gradient will be recorded for this operation.
TORCH_API Tensor uniform_(Tensor tensor, double low = 0, double high = 1);

/// Fills the input `Tensor` with values according to the method
/// described in "Understanding the difficulty of training deep feedforward
/// neural networks" - Glorot, X. & Bengio, Y. (2010). Values are scaled by the
/// `gain` parameter. No gradient will be recorded for this operation.
TORCH_API Tensor xavier_normal_(Tensor tensor, double gain = 1.0);

/// Fills the input `Tensor` with values according to the method
/// described in "Understanding the difficulty of training deep feedforward
/// neural networks" - Glorot, X. & Bengio, Y. (2010), using a uniform
/// distribution. Values are scaled by the `gain` parameter
/// No gradient will be recorded for this operation.
TORCH_API Tensor xavier_uniform_(Tensor tensor, double gain = 1.0);

/// Fills the given `tensor` with zeros.
/// No gradient will be recorded for this operation.
TORCH_API Tensor zeros_(Tensor tensor);

} // namespace init
} // namespace nn
} // namespace torch
