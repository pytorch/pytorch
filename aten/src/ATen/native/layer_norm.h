#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <c10/util/accumulate.h>

namespace at::native {

namespace {

C10_ALWAYS_INLINE void _check_rms_norm_inputs_symint(
    const Tensor& input,
    c10::SymIntArrayRef normalized_shape,
    const Tensor& weight /* optional */) {

  const int normalized_ndim = normalized_shape.size();
  TORCH_CHECK(
      normalized_ndim >= 1,
      "Expected normalized_shape to be at least 1-dimensional, i.e., ",
      "containing at least one element, but got normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      !weight.defined() || weight.sym_sizes().equals(normalized_shape),
      "Expected weight to be of same shape as normalized_shape, but got ",
      "weight of shape ",
      weight.sym_sizes(),
      " and normalized_shape = ",
      normalized_shape);

  const auto input_ndim = input.dim();
  const auto input_shape = input.sym_sizes();
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
    TORCH_CHECK(false, ss.str());
  }
}

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
    TORCH_CHECK(false, ss.str());
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

Tensor rms_norm_symint(
    const Tensor& input,
    c10::SymIntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt /* optional */,
    std::optional<double> eps);

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

DECLARE_DISPATCH(forward_fn, LayerNormKernel)
DECLARE_DISPATCH(backward_fn, LayerNormBackwardKernel)

using rmsnorm_fn = Tensor (*)(
    const Tensor& /* input */,
    c10::SymIntArrayRef /* normalized_shape */,
    const std::optional<Tensor>& /* weight_opt */,
    std::optional<double> /* eps */);
DECLARE_DISPATCH(rmsnorm_fn, RMSNormKernel)

std::tuple<Tensor, Tensor, Tensor> rms_norm_cpu(
    at::Tensor const& input,
    c10::ArrayRef<long> normalized_shape,
    std::optional<at::Tensor> const& weight_opt,
    std::optional<double> eps_opt);

std::tuple<Tensor, Tensor>
rms_norm_backward_cpu(
    const Tensor &grad,                      // Gradient with respect to the output y
    const Tensor &input,                     // Original input x
    const std::optional<Tensor> &weight,       // Optional weight (scaling factor)
    const std::optional<double> eps,         // Optional epsilon (not used here as inverse_rms is provided)
    const Tensor &output,                    // Output from the forward pass (unused in this backward computation)
    const Tensor &x_norm,                    // Normalized input (x multiplied by inverse_rms)
    const Tensor &inverse_rms,               // Inverse RMS factor: 1 / sqrt(mean(x^2) + eps)
    std::array<bool, 2ul> grad_input_mask    // Mask: [compute_grad_input, compute_grad_weight]
);

} // namespace at::native
