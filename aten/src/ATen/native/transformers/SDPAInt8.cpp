#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/grad_mode.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/transformers/SDPAInt8.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_like.h>
#include <ATen/ops/_scaled_dot_product_int8.h>
#include <ATen/ops/_scaled_dot_product_int8_native.h>
#include <ATen/ops/clamp_max.h>
#include <ATen/ops/clamp_min.h>
#include <ATen/ops/round.h>
#include <ATen/ops/softmax.h>
#include <ATen/ops/linear_native.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/matmul_native.h>
#include <ATen/ops/all.h>
#endif

namespace at::native {

DEFINE_DISPATCH(sdpa_int8_kernel);

at::Tensor sdpa_int8_math_impl(
    const at::Tensor& query_,
    const at::Tensor& key,
    const at::Tensor& value,
    double dropout_p,
    bool is_causal,
    c10::optional<at::Tensor> attn_mask_,
    c10::optional<double> scale,
    int32_t q_zp,
    float q_scale,
    int32_t k_zp,
    float k_scale,
    int32_t v_zp,
    float v_scale,
    int32_t a_zp,
    float a_scale,
    int32_t o_zp,
    float o_scale) {
  // dequant q/k/v
  auto q = (query_.to(at::kFloat) - q_zp) * q_scale;
  auto k = (key.to(at::kFloat) - k_zp) * k_scale;
  auto v = (value.to(at::kFloat) - v_zp) * v_scale;
  auto attn_mask = attn_mask_;
  if (attn_mask.has_value()) {
    *attn_mask = (*attn_mask).to(at::kFloat);
  }
  // Scale q, k before matmul for stability see https://tinyurl.com/sudb9s96 for math
  bool is_negative_scaling = scale.has_value() && scale.value() < 0.0;
  const auto scaling_factor = sdp::calculate_scale(q, is_negative_scaling ? std::abs(scale.value()) : scale).sqrt();
  q = q * (is_negative_scaling ? c10::SymFloat(0.0) - scaling_factor: scaling_factor);
  auto attn = at::matmul(q, k.transpose(-2, -1) * scaling_factor);
  if (attn_mask.has_value()) {
    if (at::areAnyTensorSubclassLike({attn, *attn_mask})) {
      attn = attn.add(*attn_mask);
    } else {
      attn.add_(*attn_mask);
    }
  }
  attn = at::softmax(attn, -1);
  // quant attn
  attn = at::clamp_max(
      at::clamp_min(at::round(attn / a_scale) + a_zp, 0), 255
  );
  // dequant attn
  attn = (attn - a_zp) * a_scale;
  auto output = at::matmul(attn, v);
  // quant output
  output = at::clamp_max(
      at::clamp_min(at::round(output / o_scale) + o_zp, 0), 255
  ).to(at::kByte);
  return output;
}

at::Tensor _scaled_dot_product_int8_cpu(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    double dropout_p,
    bool is_causal,
    const std::optional<Tensor>& attn_mask,
    std::optional<double> scale,
    int64_t q_zp,
    double q_scale,
    int64_t k_zp,
    double k_scale,
    int64_t v_zp,
    double v_scale,
    int64_t a_zp,
    double a_scale,
    int64_t o_zp,
    double o_scale) {
  const auto dtype = query.scalar_type();
  TORCH_CHECK(!query.is_nested() && !key.is_nested() && !value.is_nested(),
    "_scaled_dot_product_int8_cpu: Only accept plain inputs");
  TORCH_CHECK(!is_causal,
    "_scaled_dot_product_int8_cpu: is_causal not supported.");
  TORCH_CHECK(dtype == ScalarType::Byte,
    "_scaled_dot_product_int8_cpu: Expected data type be U8, but got ", dtype, " instead.");
  TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
    "_scaled_dot_product_int8_cpu: Accept only 4 dims inputs shape of {B, H, T, K}");
  TORCH_CHECK(dropout_p == 0.0,
    "_scaled_dot_product_int8_cpu: Currently do not support dropout > 0");
  TORCH_CHECK((query.size(3) == value.size(3)) && (key.size(3) == value.size(3)),
    "_scaled_dot_product_int8_cpu: Q/K/V should have the same head size");
  TORCH_CHECK(!attn_mask.has_value() ||
          attn_mask.value().scalar_type() == at::kFloat ||
          attn_mask.value().scalar_type() == at::kBFloat16,
    "_scaled_dot_product_int8_cpu: Expected attention mask be float or bf16");
  TORCH_CHECK(!attn_mask.has_value() ||
          (attn_mask.value().dim() == 2 || attn_mask.value().dim() == 4),
    "_scaled_dot_product_int8_cpu: Attention mask dim in {2, 4}");

  // fallback math path
  at::Tensor output = sdpa_int8_math_impl(query, key, value,
    dropout_p, is_causal, attn_mask, scale,
    q_zp, q_scale,
    k_zp, k_scale,
    v_zp, v_scale,
    a_zp, a_scale,
    o_zp, o_scale);

  // TODO @Valentine233: add flash attention int8 impl

  return output;
}

} // namespace at::native
