#include <ATen/ATen.h>
#include <ATen/native/cpu/layer_norm_kernel.h>

namespace at {
namespace native {

std::tuple<Tensor, Tensor, Tensor> native_layer_norm_cpu(
    const Tensor& input,
    const Tensor& weight /* optional */,
    const Tensor& bias /* optional */,
    int64_t M,
    int64_t N,
    double eps)
{
  Tensor Y = at::native::empty_like(input);
  Tensor mean = at::empty({M}, input.options());
  Tensor rstd = at::empty({M}, input.options());
  LayerNormKernel(kCPU, input, weight, bias, M, N, eps, &Y, &mean, &rstd);
  return std::make_tuple(Y, mean, rstd);
}

std::tuple<Tensor, Tensor, Tensor> native_layer_norm_backward_cpu(
    const Tensor& grad_out,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& weight,
    const Tensor& bias,
    int64_t M,
    int64_t N,
    double eps,
    std::array<bool, 3> grad_input_mask)
{
  Tensor dinput;
  Tensor dweight;
  Tensor dbias;
  if (grad_input_mask[0]) {
    dinput = at::native::empty_like(input);
  }
  if (grad_input_mask[1]) {
    dweight = at::native::empty_like(weight);
  }
  if (grad_input_mask[2]) {
    dbias = at::native::empty_like(weight);
  }
  LayerNormBackwardKernel(
      kCPU, grad_out, input, mean, rstd, weight, M, N, &dinput, &dweight, &dbias);
  return std::make_tuple(dinput, dweight, dbias);
}

// TODO(yangxm): Change this function to Aten impl so that we can support higher
// order gradients.
std::tuple<Tensor, Tensor, Tensor> native_layer_norm_double_backward_cpu(
    const Tensor& ddinput,
    const Tensor& ddweight,
    const Tensor& ddbias,
    const Tensor& grad_out,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& weight,
    int64_t M,
    int64_t N,
    std::array<bool, 3> grad_input_mask)
{
  Tensor dgrad_out;
  Tensor dinput;
  Tensor dweight;
  if (grad_input_mask[0]) {
    dgrad_out = at::native::empty_like(grad_out);
  }
  if (grad_input_mask[1]) {
    dinput = at::native::empty_like(input);
  }
  if (grad_input_mask[2]) {
    dweight = at::native::empty_like(weight);
  }
  LayerNormDoubleBackwardKernel(
      kCPU,
      ddinput,
      ddweight,
      ddbias,
      grad_out,
      input,
      mean,
      rstd,
      weight,
      M,
      N,
      &dgrad_out,
      &dinput,
      &dweight);
  return std::make_tuple(dgrad_out, dinput, dweight);
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

  return std::get<0>(at::native_layer_norm(input.contiguous(), weight.contiguous(), bias.contiguous(), M, N, eps));
}

DEFINE_DISPATCH(LayerNormKernel);
DEFINE_DISPATCH(LayerNormBackwardKernel);
DEFINE_DISPATCH(LayerNormDoubleBackwardKernel);

} // namespace native
} // namespace at
