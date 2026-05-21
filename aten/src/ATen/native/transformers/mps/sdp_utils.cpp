#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/transformers/mps/sdp_utils.h>

#include <ATen/Context.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_fused_sdp_choice_native.h>
#endif

#include <array>

namespace at::native::mps {
namespace {

constexpr std::array<at::ScalarType, 3> supported_dtypes{at::kFloat, at::kHalf, at::kBFloat16};

inline bool check_head_dim(const sdp::sdp_params& params, bool debug) {
  auto qd = params.query.sym_size(-1).maybe_as_int();
  auto vd = params.value.sym_size(-1).maybe_as_int();
  if (!qd.has_value() || !vd.has_value() || *qd != *vd || !prefill_attention_supports_head_dim(*qd)) {
    if (debug) {
      TORCH_WARN(
          "MPS SDPA: head_dim must match between Q and V and be one of "
          "{32, 64, 72, 80, 96, 128, 256}.");
    }
    return false;
  }
  return true;
}

inline bool check_min_rank(const sdp::sdp_params& params, bool debug) {
  // MPS lifts 3D inputs to 4D via ensure_4d, but the shared shape/head helpers
  // index sym_size(-3) so we still need rank >= 3.
  if (params.query.dim() < 3 || params.key.dim() < 3 || params.value.dim() < 3) {
    if (debug) {
      TORCH_WARN("MPS SDPA requires query, key, value to be at least 3-dimensional.");
    }
    return false;
  }
  return true;
}

} // namespace

bool can_use_flash_attention(const sdp::sdp_params& params, bool debug) {
  if (sdp::input_requires_grad(params)) {
    if (debug) {
      TORCH_WARN("Flash SDPA on MPS is forward-only; falling back when inputs require grad.");
    }
    return false;
  }
  constexpr auto constraints = std::array<bool (*)(const sdp::sdp_params&, bool), 7>{
      sdp::check_runtime_disabled_flash,
      sdp::check_nested_tensor,
      sdp::check_for_attn_mask,
      sdp::check_for_dropout,
      check_min_rank,
      sdp::check_nonzero_sequence_lengths_dense,
      sdp::check_last_dim_stride_equals_1_dense<true /*ignore_singleton_dim*/>,
  };
  for (const auto& c : constraints) {
    if (!c(params, debug)) {
      return false;
    }
  }
  if (!sdp::check_tensor_dtype(params, supported_dtypes, debug)) {
    return false;
  }
  if (!sdp::check_batch_size_and_num_heads_dense<true /*supports_gqa*/>(params, debug)) {
    return false;
  }
  return check_head_dim(params, debug);
}

bool can_use_mem_efficient_attention(const sdp::sdp_params& params, bool debug) {
  if (sdp::input_requires_grad(params)) {
    if (debug) {
      TORCH_WARN("Efficient SDPA on MPS is forward-only; falling back when inputs require grad.");
    }
    return false;
  }
  constexpr auto constraints = std::array<bool (*)(const sdp::sdp_params&, bool), 6>{
      sdp::check_runtime_disabled_mem_efficient,
      sdp::check_nested_tensor,
      sdp::check_for_dropout,
      check_min_rank,
      sdp::check_nonzero_sequence_lengths_dense,
      sdp::check_last_dim_stride_equals_1_dense<true /*ignore_singleton_dim*/>,
  };
  for (const auto& c : constraints) {
    if (!c(params, debug)) {
      return false;
    }
  }
  if (!sdp::check_tensor_dtype(params, supported_dtypes, debug)) {
    return false;
  }
  if (!sdp::check_batch_size_and_num_heads_dense<true /*supports_gqa*/>(params, debug)) {
    return false;
  }
  if (!check_head_dim(params, debug)) {
    return false;
  }
  if (params.attn_mask.has_value()) {
    auto mask_dtype = params.attn_mask.value().dtype();
    if (mask_dtype != at::kBool && mask_dtype != params.query.dtype()) {
      if (debug) {
        TORCH_WARN("Efficient SDPA on MPS: attn_mask dtype must be bool or match query dtype.");
      }
      return false;
    }
    // Vector kernels (qL <= 8) don't support is_causal + mask; only prefill does.
    auto qL_sym = params.query.sym_size(-2).maybe_as_int();
    if (params.is_causal && qL_sym.has_value() && *qL_sym <= 8) {
      if (debug) {
        TORCH_WARN("Efficient SDPA on MPS: vector kernels do not support is_causal + mask for short Q.");
      }
      return false;
    }
  }
  return true;
}

sdp::SDPBackend select_sdp_backend(const sdp::sdp_params& params) {
  auto& ctx = at::globalContext();
  if (!ctx.userEnabledMathSDP() && !ctx.userEnabledFlashSDP() && !ctx.userEnabledMemEfficientSDP()) {
    return sdp::SDPBackend::error;
  }
  for (auto backend : ctx.sDPPriorityOrder()) {
    switch (backend) {
      case sdp::SDPBackend::flash_attention:
        if (can_use_flash_attention(params, /*debug=*/false)) {
          return sdp::SDPBackend::flash_attention;
        }
        break;
      case sdp::SDPBackend::efficient_attention:
        if (can_use_mem_efficient_attention(params, /*debug=*/false)) {
          return sdp::SDPBackend::efficient_attention;
        }
        break;
      case sdp::SDPBackend::math:
        if (ctx.userEnabledMathSDP()) {
          return sdp::SDPBackend::math;
        }
        break;
      case sdp::SDPBackend::cudnn_attention:
      case sdp::SDPBackend::overrideable:
        break;
      default:
        TORCH_CHECK(false, "Invalid backend");
    }
  }
  // Re-run gates in debug to print why each was rejected. Skip backends the
  // user explicitly disabled (no point telling them their own setting).
  if (ctx.userEnabledMemEfficientSDP()) {
    TORCH_WARN("Memory-efficient SDPA on MPS not used because:");
    can_use_mem_efficient_attention(params, /*debug=*/true);
  }
  if (ctx.userEnabledFlashSDP()) {
    TORCH_WARN("Flash SDPA on MPS not used because:");
    can_use_flash_attention(params, /*debug=*/true);
  }
  return sdp::SDPBackend::error;
}

} // namespace at::native::mps

namespace at::native {

int64_t _fused_sdp_choice_mps(const Tensor& query_,
                              const Tensor& key,
                              const Tensor& value,
                              const std::optional<Tensor>& attn_mask_,
                              double dropout_p,
                              bool is_causal,
                              std::optional<double> scale,
                              bool enable_gqa) {
  sdp::sdp_params kernel_params{query_, key, value, attn_mask_, dropout_p, is_causal, enable_gqa};
  auto backend = mps::select_sdp_backend(kernel_params);
  if (backend == sdp::SDPBackend::error) {
    TORCH_CHECK(false,
                "No viable backend for scaled_dot_product_attention was found. ",
                "This is likely due to turning off the flash_attention, efficient_attention, and math kernels.");
  }
  return static_cast<int64_t>(backend);
}

REGISTER_MPS_DISPATCH(_fused_sdp_choice_stub, &_fused_sdp_choice_mps)

} // namespace at::native
