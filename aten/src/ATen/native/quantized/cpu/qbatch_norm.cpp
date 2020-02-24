#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/quantized/cpu/quantized_ops.h>

#include <algorithm>
#include <vector>

namespace at {
namespace native {

DEFINE_DISPATCH(qbatch_norm_stub);

namespace {
void compute_fused_params(
    const int64_t channels,
    const float* weight_data,
    const float* bias_data,
    const float* mean_data,
    const float* var_data,
    double eps,
    float input_scale,
    float output_scale,
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
Tensor q_batch_norm_impl(
    Tensor qx,
    Tensor weight,
    Tensor bias,
    Tensor mean,
    Tensor var,
    double eps,
    float output_scale,
    int64_t output_zero_point) {

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

  const float* weight_data = weight.template data<float>();
  const float* bias_data = bias.template data<float>();

  TORCH_CHECK(mean.numel() == C, "Mean size must match channel dimension");
  TORCH_CHECK(var.numel() == C, "Variance size must match channel dimension");

  Tensor alpha = at::empty_like(mean, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor beta = at::empty_like(mean, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  float* alpha_data = alpha.data_ptr<float>();
  float* beta_data = beta.data_ptr<float>();

  const float* mean_data = mean.template data<float>();
  const float* var_data = var.template data<float>();

  auto oSizes = qx.sizes();
  auto qx_nhwc = qx.contiguous(MemoryFormat::ChannelsLast);
  Tensor qy = at::_empty_affine_quantized(
      oSizes,
      at::device(kCPU)
         .dtype(qx_nhwc.scalar_type())
         .memory_format(MemoryFormat::ChannelsLast),
      output_scale,
      output_zero_point);

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
  qy = q_batch_norm_impl<false>(
      qx, weight, bias, mean, var, eps, output_scale, output_zero_point);
  return qy;
}

// Keep the registry in the anonymous namespace.
namespace {
class QBatchNorm2d final : public torch::OperatorKernel {
 public:
  Tensor operator()(
      Tensor qx,
      Tensor weight,
      Tensor bias,
      Tensor mean,
      Tensor var,
      double eps,
      double output_scale,
      int64_t output_zero_point) {
    return q_batch_norm_impl<false>(
        qx, weight, bias, mean, var, eps, output_scale, output_zero_point);
  }
};

static auto registry = torch::RegisterOperators().op(
    "quantized::batch_norm(Tensor qx, "
    "Tensor weight, "
    "Tensor bias, "
    "Tensor mean, "
    "Tensor var, "
    "float eps, "
    "float output_scale, "
    "int output_zero_point) -> Tensor",
    torch::RegisterOperators::options().kernel<QBatchNorm2d>(
        DispatchKey::QuantizedCPUTensorId));

} // namespace
} // namespace native
} // namespace at
