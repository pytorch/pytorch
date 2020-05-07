#include <ATen/native/layer_norm.h>

#include <array>
#include <functional>
#include <numeric>
#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

namespace at {
namespace native {

std::tuple<Tensor, Tensor, Tensor> layer_norm_cpu(
    const Tensor& X,
    const Tensor& gamma /* optional */,
    const Tensor& beta /* optional */,
    int64_t M,
    int64_t N,
    double eps) {
  Tensor Y = at::native::empty_like(X, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor mean = at::empty({M}, X.options());
  Tensor rstd = at::empty({M}, X.options());
  if (M > 0) {
    LayerNormKernel(kCPU, X, gamma, beta, M, N, eps, &Y, &mean, &rstd);
  }
  return std::make_tuple(std::move(Y), std::move(mean), std::move(rstd));
}

std::tuple<Tensor, Tensor, Tensor> layer_norm_backward_cpu(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t M,
    int64_t N,
    std::array<bool, 3> grad_input_mask) {
  Tensor dX;
  Tensor dgamma;
  Tensor dbeta;
  if (grad_input_mask[0]) {
    dX = at::native::empty_like(X, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  if (grad_input_mask[1]) {
    dgamma = M > 0 ? at::native::empty_like(gamma, LEGACY_CONTIGUOUS_MEMORY_FORMAT) : at::native::zeros_like(gamma, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  if (grad_input_mask[2]) {
    dbeta = M > 0 ? at::native::empty_like(gamma, LEGACY_CONTIGUOUS_MEMORY_FORMAT) : at::native::zeros_like(gamma, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  if (M > 0) {
    LayerNormBackwardKernel(
        kCPU, dY, X, mean, rstd, gamma, M, N, &dX, &dgamma, &dbeta);
  }
  return std::make_tuple(std::move(dX), std::move(dgamma), std::move(dbeta));
}

std::tuple<Tensor, Tensor, Tensor, int64_t, int64_t> _prepare_layer_norm_inputs(
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

  const auto& X = input.is_contiguous() ? input : input.contiguous();
  const auto& gamma = weight.is_contiguous() ? weight : weight.contiguous();
  const auto& beta = bias.is_contiguous() ? bias : bias.contiguous();

  return std::make_tuple(X, gamma, beta, M, N);
}

Tensor layer_norm(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& weight /* optional */,
    const Tensor& bias /* optional */,
    double eps,
    bool /* cudnn_enable, deprecated */) {

  auto inputs = _prepare_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto X = std::get<0>(inputs);
  auto gamma = std::get<1>(inputs);
  auto beta = std::get<2>(inputs);
  auto M = std::get<3>(inputs);
  auto N = std::get<4>(inputs);

  return std::get<0>(at::native_layer_norm(X, gamma, beta, M, N, eps));
}

Tensor quantized_layer_norm_impl(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& weight /* optional */,
    const Tensor& bias /* optional */,
    double eps,
    double output_scale,
    int64_t output_zero_point) {

  auto inputs = _prepare_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto X = std::get<0>(inputs);
  auto gamma = std::get<1>(inputs);
  auto beta = std::get<2>(inputs);
  auto M = std::get<3>(inputs);
  auto N = std::get<4>(inputs);

  Tensor Y = at::_empty_affine_quantized(
    X.sizes(),
    X.scalar_type(),
    output_scale,
    output_zero_point,
    X.suggest_memory_format());

  if (M > 0) {
    bool affine_per_channel = false;
    int num_channels = 1; // not relevant for LayerNorm
    int num_groups = 1; // not relevant for LayerNorm
    quantized_normalize_stub(kCPU, X, gamma, beta, affine_per_channel,
        num_channels, num_groups, M, N, eps, &Y);
  }
  return Y;
}

Tensor quantized_group_norm_impl(
    const Tensor& qx,
    int64_t num_groups,
    const Tensor& weight, // optional
    const Tensor& bias, // optional
    double eps,
    double output_scale,
    int64_t output_zero_point) {

  const auto input_ndim = qx.dim();
  TORCH_CHECK(
      input_ndim >= 3,
      "Expected normalized_shape to be at least 3-dimensional");
  TORCH_CHECK(num_groups > 0, "Expected num_groups to be positive");

  const auto input_shape = qx.sizes();
  TORCH_CHECK(input_shape[1] % num_groups == 0,
      "Expected channels to be divisible by groups");

  const int64_t batches = input_shape[0];
  const int64_t num_channels = input_shape[1];
  const int64_t elements_per_batch = std::accumulate(
      input_shape.cbegin() + 1,
      input_shape.cend(),
      1LL,
      std::multiplies<int64_t>());

  const int64_t M = batches * num_groups;
  const int64_t N = elements_per_batch / num_groups;

  const auto& qx_contig = qx.is_contiguous() ? qx : qx.contiguous();
  const auto& weight_contig = weight.is_contiguous() ? weight : weight.contiguous();
  const auto& bias_contig = bias.is_contiguous() ? bias : bias.contiguous();

  Tensor Y = at::_empty_affine_quantized(
    qx.sizes(),
    qx.scalar_type(),
    output_scale,
    output_zero_point,
    qx.suggest_memory_format());

  if (M > 0) {
    bool affine_per_channel = true;
    quantized_normalize_stub(kCPU, qx_contig, weight_contig, bias_contig,
        affine_per_channel, num_channels, num_groups, M, N, eps, &Y);
  }
  return Y;
}

Tensor quantized_instance_norm_impl(
    const Tensor& qx,
    const Tensor& weight, // optional
    const Tensor& bias, // optional
    double eps,
    double output_scale,
    int64_t output_zero_point) {

  const auto input_ndim = qx.dim();
  TORCH_CHECK(
      input_ndim >= 3,
      "Expected normalized_shape to be at least 3-dimensional");
  const auto input_shape = qx.sizes();

  // IN is GN with num_groups == num_channels
  const auto num_channels = input_shape[1];
  TORCH_CHECK(num_channels > 0, "Expected 2nd dimension to be positive");

  return quantized_group_norm_impl(
      qx, num_channels, weight, bias, eps, output_scale, output_zero_point);
}

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  // TODO: this is kind of... blegh
  m.impl("layer_norm", [](
    Tensor input,
    std::vector<int64_t> normalized_shape,  // because IntArrayRef doesn't work
    Tensor weight /* optional */,
    Tensor bias /* optional */,
    double eps,
    double output_scale,
    int64_t output_zero_point) {
      return quantized_layer_norm_impl(input, normalized_shape, weight, bias, eps, output_scale, output_zero_point);
  });
  m.impl("group_norm", [](
      Tensor qx,
      int64_t num_groups,
      Tensor weight,
      Tensor bias,
      double eps,
      double output_scale,
      int64_t output_zero_point) {
    return quantized_group_norm_impl(
        qx, num_groups, weight, bias, eps, output_scale, output_zero_point);
  });
  m.impl("instance_norm", [](
      Tensor qx,
      Tensor weight,
      Tensor bias,
      double eps,
      double output_scale,
      int64_t output_zero_point) {
    return quantized_instance_norm_impl(
        qx, weight, bias, eps, output_scale, output_zero_point);
  });
}

DEFINE_DISPATCH(LayerNormKernel);
DEFINE_DISPATCH(LayerNormBackwardKernel);
DEFINE_DISPATCH(quantized_normalize_stub);

} // namespace native
} // namespace at
