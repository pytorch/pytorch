#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <c10/util/accumulate.h>

namespace at::native {

namespace {

C10_ALWAYS_INLINE std::pair<int64_t, int64_t> _check_layer_norm_inputs(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& weight /* optional */,
    const Tensor& bias /* optional */) {

  const int normalized_ndim = normalized_shape.size();
  TORCH_CHECK(
      normalized_ndim >= 1,
      "Expected normalized_shape to be at least 1-dimensional, i.e., ",
      "containing at least one element, but got normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      !weight.defined() || weight.sizes().equals(normalized_shape),
      "Expected weight to be of same shape as normalized_shape, but got ",
      "weight of shape ",
      weight.sizes(),
      " and normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      !bias.defined() || bias.sizes().equals(normalized_shape),
      "Expected bias to be of same shape as normalized_shape, but got ",
      "bias of shape ",
      bias.sizes(),
      " and normalized_shape = ",
      normalized_shape);

  const auto input_shape = input.sizes();
  const auto input_ndim = input.dim();

  if (input_ndim < normalized_ndim ||
      !input_shape.slice(input_ndim - normalized_ndim)
           .equals(normalized_shape)) {
    std::stringstream ss;
    ss << "Given normalized_shape=" << normalized_shape
       << ", expected input with shape [*";
    for (auto size : normalized_shape) {
      ss << ", " << size;
    }
    ss << "], but got input of size" << input_shape;
    AT_ERROR(ss.str());
  }

  const int axis = input_ndim - normalized_ndim;
  const int64_t M =
      c10::multiply_integers(input_shape.cbegin(), input_shape.cbegin() + axis);
  const int64_t N =
      c10::multiply_integers(input_shape.cbegin() + axis, input_shape.cend());

  return std::make_pair(M, N);
}

} // namespace

void layer_norm_cpu_out(
    at::Tensor& out,
    const at::Tensor& input,
    const Tensor& gamma,
    const Tensor& beta,
    double eps,
    int64_t M,
    int64_t N);

using forward_fn = void (*)(
    const Tensor& /* X */,
    const Tensor& /* gamma */,
    const Tensor& /* beta */,
    int64_t /* M */,
    int64_t /* N */,
    double /* eps */,
    Tensor* /* Y */,
    Tensor* /* mean */,
    Tensor* /* rstd */);

using backward_fn = void (*)(
    const Tensor& /* dY */,
    const Tensor& /* X */,
    const Tensor& /* mean */,
    const Tensor& /* rstd */,
    const Tensor& /* gamma */,
    int64_t /* M */,
    int64_t /* N */,
    Tensor* /* dX */,
    Tensor* /* dgamma */,
    Tensor* /* dbeta */);

DECLARE_DISPATCH(forward_fn, LayerNormKernel);
DECLARE_DISPATCH(backward_fn, LayerNormBackwardKernel);

} // namespace at::native
