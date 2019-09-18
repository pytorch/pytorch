#include <ATen/ATen.h>
#include <ATen/native/cpu/layer_norm_kernel.h>

namespace at {
namespace native {

std::tuple<Tensor, Tensor, Tensor> native_layer_norm_cpu(
    const Tensor& input,
    const Tensor& weight /* optional */,
    const Tensor& bias /* optional */,
    int64_t normalized_ndim,
    int64_t M,
    int64_t N,
    double eps)
{
  Tensor out = at::empty_like(input);
  Tensor mean = at::empty({M}, input.options());
  Tensor rstd = at::empty({M}, input.options());
  LayerNormKernel(kCPU, input, weight, bias, M, N, eps, &out, &mean, &rstd);
  return std::make_tuple(out, mean, rstd);
}

std::tuple<Tensor, Tensor, Tensor> non_differentiable_native_layer_norm_backward_cpu(
    const Tensor& input,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& grad_out,
    const Tensor& weight,
    int64_t M,
    int64_t N,
    double eps,
    std::array<bool, 3> grad_input_mask)
{
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;
  if (grad_input_mask[0]) {
    grad_input = at::native::empty_like(input);
  }
  if (grad_input_mask[1]) {
    grad_weight = at::native::empty_like(weight);
  }
  if (grad_input_mask[2]) {
    grad_bias = at::native::empty_like(weight);
  }
  LayerNormBackwardKernel(
      kCPU, grad_out, input, mean, rstd, weight, M, N, &grad_input, &grad_weight, &grad_bias);
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

Tensor layer_norm(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& weight /* optional */,
    const Tensor& bias /* optional */,
    double eps)
{
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
  const int64_t M = std::accumulate(
      input_shape.cbegin(),
      input_shape.cbegin() + axis,
      1LL,
      std::multiplies<int64_t>());
  const int64_t N = std::accumulate(
      input_shape.cbegin() + axis,
      input_shape.cend(),
      1LL,
      std::multiplies<int64_t>());

  return std::get<0>(at::native_layer_norm(input.contiguous(), weight.contiguous(), bias.contiguous(), normalized_ndim, M, N, eps));
}

DEFINE_DISPATCH(LayerNormKernel);
DEFINE_DISPATCH(LayerNormBackwardKernel);

} // namespace native
} // namespace at
