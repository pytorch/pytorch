#pragma once

#include <torch/csrc/Export.h>
#include <torch/enum.h>
#include <torch/types.h>

namespace torch {

namespace nn::init {

using NonlinearityType = std::variant<
    enumtype::kLinear,
    enumtype::kConv1D,
    enumtype::kConv2D,
    enumtype::kConv3D,
    enumtype::kConvTranspose1D,
    enumtype::kConvTranspose2D,
    enumtype::kConvTranspose3D,
    enumtype::kSigmoid,
    enumtype::kTanh,
    enumtype::kReLU,
    enumtype::kLeakyReLU>;

using FanModeType = std::variant<enumtype::kFanIn, enumtype::kFanOut>;

} // namespace nn::init

namespace nn::init {

/// Return the recommended gain value for the given nonlinearity function.
TORCH_API double calculate_gain(
    NonlinearityType nonlinearity,
    double param = 0.01);

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
/// described in "Delving deep into rectifiers: Surpassing human-level
/// performance on ImageNet classification" - He, K. et al. (2015), using a
/// normal distribution. Also known as He initialization.
/// No gradient will be recorded for this operation.
TORCH_API Tensor kaiming_normal_(
    Tensor tensor,
    double a = 0,
    FanModeType mode = torch::kFanIn,
    NonlinearityType nonlinearity = torch::kLeakyReLU);

/// Fills the input `Tensor` with values according to the method
/// described in "Delving deep into rectifiers: Surpassing human-level
/// performance on ImageNet classification" - He, K. et al. (2015), using a
/// uniform distribution. Also known as He initialization.
/// No gradient will be recorded for this operation.
TORCH_API Tensor kaiming_uniform_(
    Tensor tensor,
    double a = 0,
    FanModeType mode = torch::kFanIn,
    NonlinearityType nonlinearity = torch::kLeakyReLU);

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

TORCH_API std::tuple<int64_t, int64_t> _calculate_fan_in_and_fan_out(
    const Tensor& tensor);

} // namespace nn::init

} // namespace torch
