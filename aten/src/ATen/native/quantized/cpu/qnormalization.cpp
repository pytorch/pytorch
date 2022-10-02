#include <ATen/core/Tensor.h>
#include <ATen/native/layer_norm.h>
#include <ATen/native/quantized/cpu/QuantizedOps.h>
#include <ATen/Parallel.h>
#include <c10/util/accumulate.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#endif

#include <algorithm>
#include <vector>

namespace at {
namespace native {

DEFINE_DISPATCH(quantized_normalize_stub);
DEFINE_DISPATCH(quantized_groupnorm_nhwc_stub);

Tensor quantized_layer_norm_impl(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& weight /* optional */,
    const Tensor& bias /* optional */,
    double eps,
    double output_scale,
    int64_t output_zero_point) {

  auto M_N = _check_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto M = M_N.first;
  auto N = M_N.second;
  auto X = input.expect_contiguous();
  auto gamma = weight.expect_contiguous();
  auto beta = bias.expect_contiguous();

  Tensor Y = at::_empty_affine_quantized(
    X->sizes(),
    X->scalar_type(),
    output_scale,
    output_zero_point,
    X->suggest_memory_format());

  if (M > 0) {
    bool affine_per_channel = false;
    int num_channels = 1; // not relevant for LayerNorm
    int num_groups = 1; // not relevant for LayerNorm
    quantized_normalize_stub(kCPU, *X, *gamma, *beta, affine_per_channel,
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
  bool is_channels_last = qx.is_contiguous(c10::MemoryFormat::ChannelsLast);
  auto mem_layout = is_channels_last ? c10::MemoryFormat::ChannelsLast :
                                       c10::MemoryFormat::Contiguous;

  const auto& qx_contig = qx.contiguous(mem_layout);
  const auto& weight_contig = weight.contiguous();
  const auto& bias_contig = bias.contiguous();

  const auto input_ndim = qx_contig.dim();
  TORCH_CHECK(
      input_ndim >= 3,
      "Expected normalized_shape to be at least 3-dimensional");
  TORCH_CHECK(num_groups > 0, "Expected num_groups to be positive");

  const auto input_shape = qx_contig.sizes();
  TORCH_CHECK(input_shape[1] % num_groups == 0,
      "Expected channels to be divisible by groups");

  const int64_t batches = input_shape[0];
  const int64_t num_channels = input_shape[1];
  const int64_t elements_per_batch =
      c10::multiply_integers(input_shape.cbegin() + 1, input_shape.cend());

  const int64_t M = batches * num_groups;
  const int64_t N = elements_per_batch / num_groups;

  Tensor Y = at::_empty_affine_quantized(
    qx_contig.sizes(),
    qx_contig.scalar_type(),
    output_scale,
    output_zero_point,
    qx_contig.suggest_memory_format());

  if (M > 0) {
    bool affine_per_channel = true;
    if (is_channels_last) {
      quantized_groupnorm_nhwc_stub(kCPU, qx_contig, weight_contig, bias_contig,
          affine_per_channel, num_channels, num_groups, M, N, eps, &Y);
    } else {
      quantized_normalize_stub(kCPU, qx_contig, weight_contig, bias_contig,
          affine_per_channel, num_channels, num_groups, M, N, eps, &Y);
    }
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
  m.impl(TORCH_SELECTIVE_NAME("quantized::layer_norm"), [](
    Tensor input,
    std::vector<int64_t> normalized_shape,  // because IntArrayRef doesn't work
    c10::optional<Tensor> weight,
    c10::optional<Tensor> bias,
    double eps,
    double output_scale,
    int64_t output_zero_point) {
      return quantized_layer_norm_impl(
          input, normalized_shape,
          weight.has_value() ? *weight : Tensor(),
          bias.has_value() ? *bias : Tensor(),
          eps, output_scale, output_zero_point);
  });
  m.impl(TORCH_SELECTIVE_NAME("quantized::group_norm"), [](
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
  m.impl(TORCH_SELECTIVE_NAME("quantized::instance_norm"), [](
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
