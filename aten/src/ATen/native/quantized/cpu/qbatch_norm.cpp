#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <ATen/native/quantized/cpu/quantized_ops.h>

#include <algorithm>
#include <vector>

namespace at {
namespace native {

DEFINE_DISPATCH(qbatch_norm_stub);
DEFINE_DISPATCH(qbatch_norm_relu_stub);

namespace {
void compute_fused_params(
    const int64_t channels,
    const float* weight_data,
    const float* bias_data,
    const float* mean_data,
    const float* var_data,
    double eps,
    double input_scale,
    double output_scale,
    float* alpha_data,
    float* beta_data) {
  // Batch Normalization
  // output(n, c, h, w)
  //     = (input(n, c, h, w) - mean(c)) / sqrt(var(c) + eps) * weight(c)
  //         + bias(c)
  // We factor out inv_sigma(c) = 1 / sqrt(var(c) + eps).
  for (int64_t c = 0; c < channels; c++) {
    float inv_sigma = 1.0 / std::sqrt(var_data[c] + static_cast<float>(eps));
    float weight_v = weight_data ? weight_data[c] : 1;
    float bias_v = bias_data ? bias_data[c] : 0;
    alpha_data[c] = inv_sigma * weight_v * (input_scale / output_scale);
    beta_data[c] = (bias_v - mean_data[c] * inv_sigma * weight_v) / output_scale;
  }
}

template <bool ReluFused>
Tensor q_batch_norm1d_impl(
    Tensor qx,
    c10::optional<Tensor> mb_weight,
    c10::optional<Tensor> mb_bias,
    Tensor mean,
    Tensor var,
    double eps,
    double output_scale,
    int64_t output_zero_point) {

  TORCH_CHECK(mb_weight.has_value(), "Weight must be provided");
  TORCH_CHECK(mb_bias.has_value(), "Bias must be provided");
  const auto& weight = *mb_weight;
  const auto& bias = *mb_bias;

  if (qx.numel() == 0) {
    auto out = qx.clone();
    return out;
  }
  int64_t ndim = qx.dim();
  TORCH_CHECK(ndim == 2 || ndim == 3, "Expecting the input tensor of rank 2 or 3.");
  const int64_t N = qx.size(0);
  const int64_t C = qx.size(1);
  const int64_t H = ndim == 3 ? qx.size(2) : 1;

  TORCH_CHECK(weight.numel() == C, "Expect weight size to match C");
  TORCH_CHECK(bias.numel() == C, "Expect weight size to match C");

  const float* weight_data = weight.template data_ptr<float>();
  const float* bias_data = bias.template data_ptr<float>();

  TORCH_CHECK(mean.numel() == C, "Mean size must match channel dimension");
  TORCH_CHECK(var.numel() == C, "Variance size must match channel dimension");

  Tensor alpha = at::empty_like(mean, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor beta = at::empty_like(mean, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  float* alpha_data = alpha.data_ptr<float>();
  float* beta_data = beta.data_ptr<float>();

  const float* mean_data = mean.template data_ptr<float>();
  const float* var_data = var.template data_ptr<float>();

  if (ndim == 2) {
    // create a fake H and W dimension so we can use NHWC
    qx = qx.unsqueeze(-1).unsqueeze(-1);
  } else {
    // create a fake W dimension so we can use NHWC
    qx = qx.unsqueeze(-1);
  }

  auto oSizes = qx.sizes();
  auto qx_nhwc = qx.contiguous(MemoryFormat::ChannelsLast);
  Tensor qy = at::_empty_affine_quantized(
      oSizes,
      at::device(kCPU)
        .dtype(qx_nhwc.scalar_type())
        .memory_format(MemoryFormat::ChannelsLast),
      output_scale,
      output_zero_point,
      c10::nullopt);

  compute_fused_params(
      C,
      weight_data,
      bias_data,
      mean_data,
      var_data,
      eps,
      qx.q_scale(),
      output_scale,
      alpha_data,
      beta_data);
  if (ReluFused) {
    qbatch_norm_relu_stub(
        qx.device().type(),
        N,
        C,
        H,
        qx.q_zero_point(),
        output_zero_point,
        qx_nhwc,
        alpha,
        beta,
        qy);
  } else {
    qbatch_norm_stub(
        qx.device().type(),
        N,
        C,
        H,
        qx.q_zero_point(),
        output_zero_point,
        qx_nhwc,
        alpha,
        beta,
        qy);
  }
  // Remove the fake dimension, and go back to contiguous format
  // (since there is no 4th channel). Note, this has a performance
  // cost.
  return qy.contiguous(MemoryFormat::Contiguous).squeeze(-1);
}

template <bool ReluFused>
Tensor q_batch_norm2d_impl(
    Tensor qx,
    c10::optional<Tensor> mb_weight,
    c10::optional<Tensor> mb_bias,
    Tensor mean,
    Tensor var,
    double eps,
    double output_scale,
    int64_t output_zero_point) {

  TORCH_CHECK(mb_weight.has_value(), "Weight must be provided");
  TORCH_CHECK(mb_bias.has_value(), "Bias must be provided");
  const auto& weight = *mb_weight;
  const auto& bias = *mb_bias;

  if (qx.numel() == 0) {
    auto out = qx.clone();
    return out;
  }
  int64_t ndim = qx.dim();
  TORCH_CHECK(ndim == 4, "Expecting the input tensor of rank 4.");
  const int64_t N = qx.size(0);
  const int64_t C = qx.size(1);
  const int64_t H = qx.size(2);
  const int64_t W = qx.size(3);

  TORCH_CHECK(weight.numel() == C, "Expect weight size to match C");
  TORCH_CHECK(bias.numel() == C, "Expect weight size to match C");

  const float* weight_data = weight.template data_ptr<float>();
  const float* bias_data = bias.template data_ptr<float>();

  TORCH_CHECK(mean.numel() == C, "Mean size must match channel dimension");
  TORCH_CHECK(var.numel() == C, "Variance size must match channel dimension");

  Tensor alpha = at::empty_like(mean, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor beta = at::empty_like(mean, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  float* alpha_data = alpha.data_ptr<float>();
  float* beta_data = beta.data_ptr<float>();

  const float* mean_data = mean.template data_ptr<float>();
  const float* var_data = var.template data_ptr<float>();

  auto oSizes = qx.sizes();
  auto qx_nhwc = qx.contiguous(MemoryFormat::ChannelsLast);
  Tensor qy = at::_empty_affine_quantized(
      oSizes,
      at::device(kCPU)
        .dtype(qx_nhwc.scalar_type())
        .memory_format(MemoryFormat::ChannelsLast),
      output_scale,
      output_zero_point,
      c10::nullopt);

  compute_fused_params(
      C,
      weight_data,
      bias_data,
      mean_data,
      var_data,
      eps,
      qx.q_scale(),
      output_scale,
      alpha_data,
      beta_data);
  if (ReluFused) {
    qbatch_norm_relu_stub(
        qx.device().type(),
        N,
        C,
        H * W,
        qx.q_zero_point(),
        output_zero_point,
        qx_nhwc,
        alpha,
        beta,
        qy);
  } else {
    qbatch_norm_stub(
        qx.device().type(),
        N,
        C,
        H * W,
        qx.q_zero_point(),
        output_zero_point,
        qx_nhwc,
        alpha,
        beta,
        qy);
  }
  return qy;
}

template <bool ReluFused>
Tensor q_batch_norm3d_impl(
    Tensor qx,
    c10::optional<Tensor> mb_weight,
    c10::optional<Tensor> mb_bias,
    Tensor mean,
    Tensor var,
    double eps,
    double output_scale,
    int64_t output_zero_point) {

  TORCH_CHECK(mb_weight.has_value(), "Weight must be provided")
  TORCH_CHECK(mb_bias.has_value(), "Bias must be provided")

  const auto& weight = *mb_weight;
  const auto& bias = *mb_bias;

  if (qx.numel() == 0) {
    auto out = qx.clone();
    return out;
  }
  int64_t ndim = qx.dim();
  TORCH_CHECK(ndim == 5, "Expecting the input tensor of rank 5.");
  const int64_t N = qx.size(0);
  const int64_t C = qx.size(1);
  const int64_t D = qx.size(2);
  const int64_t H = qx.size(3);
  const int64_t W = qx.size(4);

  TORCH_CHECK(weight.numel() == C, "Expect weight size to match C");
  TORCH_CHECK(bias.numel() == C, "Expect weight size to match C");

  const float* weight_data = weight.template data_ptr<float>();
  const float* bias_data = bias.template data_ptr<float>();

  TORCH_CHECK(mean.numel() == C, "Mean size must match channel dimension");
  TORCH_CHECK(var.numel() == C, "Variance size must match channel dimension");

  Tensor alpha = at::empty_like(mean, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor beta = at::empty_like(mean, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  float* alpha_data = alpha.data_ptr<float>();
  float* beta_data = beta.data_ptr<float>();

  const float* mean_data = mean.template data_ptr<float>();
  const float* var_data = var.template data_ptr<float>();

  auto oSizes = qx.sizes();
  auto qx_nhwc = qx.contiguous(MemoryFormat::ChannelsLast3d);
  Tensor qy = at::_empty_affine_quantized(
      oSizes,
      at::device(kCPU)
        .dtype(qx_nhwc.scalar_type())
        .memory_format(MemoryFormat::ChannelsLast3d),
      output_scale,
      output_zero_point,
      c10::nullopt);

  compute_fused_params(
      C,
      weight_data,
      bias_data,
      mean_data,
      var_data,
      eps,
      qx.q_scale(),
      output_scale,
      alpha_data,
      beta_data);

  if (ReluFused) {
    qbatch_norm_relu_stub(
        qx.device().type(),
        N,
        C,
        D * H * W,
        qx.q_zero_point(),
        output_zero_point,
        qx_nhwc,
        alpha,
        beta,
        qy);
  } else {
    qbatch_norm_stub(
        qx.device().type(),
        N,
        C,
        D * H * W,
        qx.q_zero_point(),
        output_zero_point,
        qx_nhwc,
        alpha,
        beta,
        qy);
  }
  return qy;
}

template <bool ReluFused>
Tensor q_batch_norm_impl(
    Tensor qx,
    c10::optional<Tensor> mb_weight,
    c10::optional<Tensor> mb_bias,
    Tensor mean,
    Tensor var,
    double eps,
    double output_scale,
    int64_t output_zero_point) {
  Tensor qy;
  int64_t dim = qx.dim();
  if (dim == 2 || dim == 3) {
    qy = q_batch_norm1d_impl<ReluFused>(
        qx, mb_weight, mb_bias, mean, var, eps, output_scale, output_zero_point);
  } else if (dim == 4) {
    qy = q_batch_norm2d_impl<ReluFused>(
        qx, mb_weight, mb_bias, mean, var, eps, output_scale, output_zero_point);
  } else if (dim == 5) {
    qy = q_batch_norm3d_impl<ReluFused>(
        qx, mb_weight, mb_bias, mean, var, eps, output_scale, output_zero_point);
  } else {
    TORCH_CHECK(false, "quantized::batch_norm only support 2d, 3d, 4d or 5d inputs.");
  }
  return qy;
}

} // namespace

Tensor quantized_batch_norm(
    const Tensor& qx,
    const Tensor& weight /* optional */,
    const Tensor& bias /* optional */,
    const Tensor& mean /* optional */,
    const Tensor& var /* optional */,
    double eps,
    double output_scale,
    int64_t output_zero_point) {
  Tensor qy;
  // TODO: this should arguably support 3d as well
  qy = q_batch_norm2d_impl<false>(
      qx,
      weight.defined() ? c10::make_optional(weight) : c10::nullopt,
      bias.defined() ? c10::make_optional(bias) : c10::nullopt,
      mean, var, eps, output_scale, output_zero_point);
  return qy;
}

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::batch_norm"),        TORCH_FN(q_batch_norm_impl<false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::batch_norm_relu"),   TORCH_FN(q_batch_norm_impl<true>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::batch_norm1d"),      TORCH_FN(q_batch_norm1d_impl<false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::batch_norm1d_relu"), TORCH_FN(q_batch_norm1d_impl<true>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::batch_norm2d"),      TORCH_FN(q_batch_norm2d_impl<false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::batch_norm2d_relu"), TORCH_FN(q_batch_norm2d_impl<true>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::batch_norm3d"),      TORCH_FN(q_batch_norm3d_impl<false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::batch_norm3d_relu"), TORCH_FN(q_batch_norm3d_impl<true>));
}

} // namespace native
} // namespace at
