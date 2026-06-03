#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ExpandUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_fused_sdp_choice_native.h>
#include <ATen/ops/_safe_softmax.h>
#include <ATen/ops/_scaled_dot_product_efficient_attention_backward_native.h>
#include <ATen/ops/add.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/sub.h>
#include <ATen/ops/zeros_like.h>
#endif

namespace at::native {

namespace {

bool is_supported_floating_dtype(ScalarType dtype) {
  return dtype == kFloat || dtype == kHalf || dtype == kBFloat16;
}

bool check_mps_efficient_attention_constraints(const Tensor& query,
                                               const Tensor& key,
                                               const Tensor& value,
                                               const std::optional<Tensor>& attn_mask,
                                               double dropout_p,
                                               bool enable_gqa,
                                               bool debug) {
  auto reject = [debug](const char* msg) {
    if (debug) {
      TORCH_WARN(msg);
    }
    return false;
  };

  if (query.device().type() != kMPS || key.device().type() != kMPS || value.device().type() != kMPS) {
    return reject("MPS efficient attention requires query, key, and value to be on MPS.");
  }
  if (query.is_nested() || key.is_nested() || value.is_nested()) {
    return reject("MPS efficient attention does not support NestedTensor inputs.");
  }
  if (query.dim() != 4 || key.dim() != 4 || value.dim() != 4) {
    return reject("MPS efficient attention requires 4D dense query, key, and value tensors.");
  }
  if (dropout_p != 0.0) {
    return reject("MPS efficient attention does not support dropout.");
  }
  if (enable_gqa) {
    return reject("MPS efficient attention does not support grouped query attention yet.");
  }
  if (!is_supported_floating_dtype(query.scalar_type()) || query.scalar_type() != key.scalar_type() ||
      query.scalar_type() != value.scalar_type()) {
    return reject("MPS efficient attention requires matching float32, float16, or bfloat16 Q/K/V dtypes.");
  }
  if (!query.is_contiguous() || !key.is_contiguous() || !value.is_contiguous()) {
    return reject("MPS efficient attention requires contiguous Q/K/V tensors.");
  }
  if (query.size(0) != key.size(0) || query.size(0) != value.size(0) || query.size(1) != key.size(1) ||
      query.size(1) != value.size(1)) {
    return reject("MPS efficient attention requires matching batch size and head count.");
  }
  if (query.size(2) != key.size(2) || key.size(2) != value.size(2)) {
    return reject("MPS efficient attention currently requires q, k, and v sequence lengths to match.");
  }
  if (query.size(3) != key.size(3) || query.size(3) != value.size(3)) {
    return reject("MPS efficient attention requires matching Q/K/V head dimensions.");
  }
  return true;
}

bool can_use_tiled_backward(const Tensor& grad_out,
                            const Tensor& query,
                            const Tensor& key,
                            const Tensor& value,
                            const Tensor& attn_bias,
                            const Tensor& out,
                            const Tensor& logsumexp,
                            double dropout_p,
                            std::array<bool, 4> grad_input_mask) {
  // attn_bias-defined and bias-gradient cases are handled by the dense math
  // fallback, which materializes the full attention matrix once and computes
  // grad_bias directly. The tiled path covers the common training case.
  if (attn_bias.defined() || grad_input_mask[3]) {
    return false;
  }
  if (!check_mps_efficient_attention_constraints(
          query, key, value, std::nullopt, dropout_p, /*enable_gqa=*/false, /*debug=*/false)) {
    return false;
  }
  return grad_out.dim() == 4 && grad_out.sizes() == query.sizes() &&
      grad_out.scalar_type() == query.scalar_type() && out.sizes() == query.sizes() && logsumexp.dim() == 3 &&
      logsumexp.size(0) == query.size(0) && logsumexp.size(1) == query.size(1) &&
      logsumexp.size(2) == query.size(2);
}

int64_t pick_tile_size(int64_t S, int64_t D) {
  // Tile width along the K/V sequence dimension. Smaller tiles cut peak memory;
  // larger tiles amortize MPSGraph dispatch overhead. 256 is a good default on
  // M-series; halve when D is large to keep the per-tile working set similar.
  const int64_t bc = (D >= 128) ? 128 : 256;
  return std::min<int64_t>(S, bc);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> tiled_backward(const Tensor& grad_out_,
                                                          const Tensor& query,
                                                          const Tensor& key,
                                                          const Tensor& value,
                                                          const Tensor& out,
                                                          const Tensor& logsumexp,
                                                          std::array<bool, 4> grad_input_mask,
                                                          bool is_causal,
                                                          std::optional<double> scale) {
  const int64_t S = query.size(2);
  const int64_t D = query.size(3);
  const auto scale_f = sdp::calculate_scale(query, scale).expect_float();
  const bool needs_promote = query.scalar_type() == kHalf || query.scalar_type() == kBFloat16;
  const ScalarType io_dtype = query.scalar_type();

  Tensor grad_out = grad_out_.is_contiguous() ? grad_out_ : grad_out_.contiguous();
  auto qa = needs_promote ? query.to(kFloat) : query;
  auto ka = needs_promote ? key.to(kFloat) : key;
  auto va = needs_promote ? value.to(kFloat) : value;
  auto goa = needs_promote ? grad_out.to(kFloat) : grad_out;
  auto outa = needs_promote ? out.to(kFloat) : out;

  // logsumexp comes from the forward in fp32; broadcast across the BC dim.
  auto lse = logsumexp.unsqueeze(-1); // [B, H, S, 1]
  // delta[i] = sum_d gO[i, d] * O[i, d]; reused across every K/V tile.
  auto delta = at::mul(goa, outa).sum(-1, /*keepdim=*/true); // [B, H, S, 1]

  const bool need_gq = grad_input_mask[0];
  const bool need_gk = grad_input_mask[1];
  const bool need_gv = grad_input_mask[2];
  Tensor grad_q = need_gq ? at::zeros_like(qa) : Tensor{};
  Tensor grad_k = need_gk ? at::zeros_like(ka) : Tensor{};
  Tensor grad_v = need_gv ? at::zeros_like(va) : Tensor{};

  const int64_t BC = pick_tile_size(S, D);
  Tensor q_idx;
  if (is_causal) {
    q_idx = at::arange(S, query.options().dtype(kLong)); // [S]
  }

  for (int64_t kv0 = 0; kv0 < S; kv0 += BC) {
    const int64_t kv1 = std::min(kv0 + BC, S);
    auto k_t = ka.slice(/*dim=*/2, kv0, kv1); // [B, H, BC, D]
    auto v_t = va.slice(/*dim=*/2, kv0, kv1); // [B, H, BC, D]
    auto s_t = at::mul(at::matmul(qa, k_t.transpose(-2, -1)), scale_f); // [B, H, S, BC]

    if (is_causal) {
      auto k_idx = at::arange(kv0, kv1, query.options().dtype(kLong)); // [BC]
      auto allowed = q_idx.unsqueeze(-1).ge(k_idx.unsqueeze(0)); // [S, BC] bool
      s_t = s_t.masked_fill(allowed.logical_not(), -std::numeric_limits<float>::infinity());
    }

    auto p_t = at::sub(s_t, lse).exp(); // [B, H, S, BC] in fp32
    auto dp_t = at::matmul(goa, v_t.transpose(-2, -1)); // [B, H, S, BC]
    auto ds_t = at::mul(p_t, at::sub(dp_t, delta)); // [B, H, S, BC]

    if (need_gv) {
      grad_v.slice(/*dim=*/2, kv0, kv1).add_(at::matmul(p_t.transpose(-2, -1), goa)); // [B, H, BC, D]
    }
    if (need_gk) {
      grad_k.slice(/*dim=*/2, kv0, kv1).add_(at::mul(at::matmul(ds_t.transpose(-2, -1), qa), scale_f));
    }
    if (need_gq) {
      grad_q.add_(at::mul(at::matmul(ds_t, k_t), scale_f)); // [B, H, S, D]
    }
  }

  if (needs_promote) {
    if (grad_q.defined()) {
      grad_q = grad_q.to(io_dtype);
    }
    if (grad_k.defined()) {
      grad_k = grad_k.to(io_dtype);
    }
    if (grad_v.defined()) {
      grad_v = grad_v.to(io_dtype);
    }
  }
  return std::make_tuple(std::move(grad_q), std::move(grad_k), std::move(grad_v), Tensor{});
}

std::tuple<Tensor, Tensor, Tensor, Tensor> math_fallback_backward(const Tensor& grad_out,
                                                                  const Tensor& query,
                                                                  const Tensor& key,
                                                                  const Tensor& value,
                                                                  const Tensor& attn_bias,
                                                                  const Tensor& out,
                                                                  double dropout_p,
                                                                  std::array<bool, 4> grad_input_mask,
                                                                  bool is_causal,
                                                                  std::optional<double> scale) {
  TORCH_CHECK(dropout_p == 0.0, "MPS efficient attention backward fallback does not support dropout");
  TORCH_CHECK(!(is_causal && attn_bias.defined()),
              "MPS efficient attention backward fallback does not support attn_bias with is_causal=True");

  const auto scale_factor = sdp::calculate_scale(query, scale).expect_float();
  const bool use_float_accum = query.scalar_type() == kHalf || query.scalar_type() == kBFloat16;
  auto q = use_float_accum ? query.to(kFloat) : query;
  auto k = use_float_accum ? key.to(kFloat) : key;
  auto v = use_float_accum ? value.to(kFloat) : value;
  auto go = use_float_accum ? grad_out.to(kFloat) : grad_out;
  auto out_acc = use_float_accum ? out.to(kFloat) : out;

  auto scores = at::mul(at::matmul(q, k.transpose(-2, -1)), scale_factor);
  if (is_causal) {
    auto causal_mask = at::ones({query.size(-2), key.size(-2)}, query.options().dtype(kBool)).tril();
    scores = scores.masked_fill(causal_mask.logical_not(), -std::numeric_limits<float>::infinity());
  } else if (attn_bias.defined()) {
    scores = at::add(scores, attn_bias.to(scores.scalar_type()));
  }

  auto p = at::_safe_softmax(scores, -1);
  auto delta = at::mul(go, out_acc).sum(-1, /*keepdim=*/true);
  auto dp = at::matmul(go, v.transpose(-2, -1));
  auto ds = at::mul(p, at::sub(dp, delta));

  Tensor grad_q;
  Tensor grad_k;
  Tensor grad_v;
  Tensor grad_bias;
  if (grad_input_mask[0]) {
    grad_q = at::mul(at::matmul(ds, k), scale_factor).to(query.scalar_type());
  }
  if (grad_input_mask[1]) {
    grad_k = at::mul(at::matmul(ds.transpose(-2, -1), q), scale_factor).to(key.scalar_type());
  }
  if (grad_input_mask[2]) {
    grad_v = at::matmul(p.transpose(-2, -1), go).to(value.scalar_type());
  }
  if (grad_input_mask[3] && attn_bias.defined()) {
    grad_bias = at::sum_to(ds, attn_bias.sym_sizes()).to(attn_bias.scalar_type());
  }
  return std::make_tuple(std::move(grad_q), std::move(grad_k), std::move(grad_v), std::move(grad_bias));
}

int64_t fused_sdp_choice_mps_impl(const Tensor& query,
                                  const Tensor& key,
                                  const Tensor& value,
                                  const std::optional<Tensor>& attn_mask,
                                  double dropout_p,
                                  bool is_causal,
                                  std::optional<double> /*scale*/,
                                  bool enable_gqa) {
  (void)is_causal;
  auto& ctx = at::globalContext();
  const bool wants_efficient = ctx.userEnabledMemEfficientSDP();
  const bool wants_math = ctx.userEnabledMathSDP();

  // Conservative routing: the efficient path runs only when the user has
  // explicitly opted in via `sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])`
  // (i.e. math is disabled). Default behaviour on MPS is unchanged. Auto-
  // promotion would require per-SKU benchmark coverage across the M-series,
  // which is intentionally out of scope for this PR.
  if (wants_efficient && !wants_math &&
      check_mps_efficient_attention_constraints(
          query, key, value, attn_mask, dropout_p, enable_gqa, /*debug=*/false)) {
    return static_cast<int64_t>(sdp::SDPBackend::efficient_attention);
  }
  return static_cast<int64_t>(sdp::SDPBackend::math);
}

} // namespace

int64_t _fused_sdp_choice_mps(const Tensor& query,
                              const Tensor& key,
                              const Tensor& value,
                              const std::optional<Tensor>& attn_mask,
                              double dropout_p,
                              bool is_causal,
                              std::optional<double> scale,
                              bool enable_gqa) {
  return fused_sdp_choice_mps_impl(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> _scaled_dot_product_efficient_attention_backward_mps(
    const Tensor& grad_out,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& attn_bias,
    const Tensor& out,
    const Tensor& logsumexp,
    const Tensor& philox_seed,
    const Tensor& philox_offset,
    double dropout_p,
    std::array<bool, 4> grad_input_mask,
    bool is_causal,
    std::optional<double> scale) {
  (void)philox_seed;
  (void)philox_offset;
  if (!grad_out.defined()) {
    return std::make_tuple(Tensor{}, Tensor{}, Tensor{}, Tensor{});
  }
  if (can_use_tiled_backward(
          grad_out, query, key, value, attn_bias, out, logsumexp, dropout_p, grad_input_mask)) {
    return tiled_backward(grad_out, query, key, value, out, logsumexp, grad_input_mask, is_causal, scale);
  }
  return math_fallback_backward(
      grad_out, query, key, value, attn_bias, out, dropout_p, grad_input_mask, is_causal, scale);
}

REGISTER_MPS_DISPATCH(_fused_sdp_choice_stub, &_fused_sdp_choice_mps)

} // namespace at::native
