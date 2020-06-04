#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <ATen/native/quantized/cpu/quantized_ops.h>
#include <ATen/native/layer_norm.h>

#include <algorithm>
#include <vector>

namespace at {
namespace native {

DEFINE_DISPATCH(quantized_normalize_stub);

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
    c10::optional<Tensor> weight,
    c10::optional<Tensor> bias /* optional */,
    double eps,
    double output_scale,
    int64_t output_zero_point) {
      return quantized_layer_norm_impl(
          input, normalized_shape,
          weight.has_value() ? *weight : Tensor(),
          bias.has_value() ? *bias : Tensor(),
          eps, output_scale, output_zero_point);
  });
  m.impl("group_norm", [](
      Tensor qx,
      int64_t num_groups,
      c10::optional<Tensor> weight,
      c10::optional<Tensor> bias,
      double eps,
      double output_scale,
      int64_t output_zero_point) {
    return quantized_group_norm_impl(
        qx, num_groups,
        weight.has_value() ? *weight : Tensor(),
        bias.has_value() ? *bias : Tensor(),
        eps, output_scale, output_zero_point);
  });
  m.impl("instance_norm", [](
      Tensor qx,
      c10::optional<Tensor> weight,
      c10::optional<Tensor> bias,
      double eps,
      double output_scale,
      int64_t output_zero_point) {
    return quantized_instance_norm_impl(
        qx,
        weight.has_value() ? *weight : Tensor(),
        bias.has_value() ? *bias : Tensor(),
        eps, output_scale, output_zero_point);
  });
}

} // namespace native
} // namespace at
