#include <ATen/Context.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNN.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <ATen/native/transformers/xpu/sdp_utils.h>
#include <c10/util/Array.h>
#include <torch/library.h>

namespace {
bool check_head_dim_size_xpu(
    sdp::sdp_params const& params,
    c10::OptionalRef<sdp::SDPDiagnostics> diagnostics) {
  const auto query_size_last = params.query.sym_size(-1);
  const auto key_size_last = params.key.sym_size(-1);
  const auto value_size_last = params.value.sym_size(-1);
  if (query_size_last != key_size_last) {
    sdp::report_failure(
        diagnostics,
        "OneDNN attention requires q,k to have the same last dimension.",
        " Got Query.size(-1): ",
        query_size_last,
        ", Key.size(-1): ",
        key_size_last,
        " instead.");
    return false;
  }

  constexpr int MAX_HEAD_DIM = 576;
  const auto max_size_last = query_size_last.max(value_size_last);
  if (max_size_last > MAX_HEAD_DIM) {
    sdp::report_failure(
        diagnostics,
        "OneDNN attention requires q,k,v to have head dimension less than ",
        MAX_HEAD_DIM,
        ". Got ",
        max_size_last,
        " instead.");
    return false;
  }
  return true;
}

bool check_head_dim_size_xpu(sdp::sdp_params const& params, bool debug) {
  if (!debug) {
    return check_head_dim_size_xpu(
        params, c10::OptionalRef<sdp::SDPDiagnostics>{});
  }
  sdp::SDPDiagnostics diagnostics(true);
  return check_head_dim_size_xpu(params, diagnostics);
}

bool check_no_grad(
    sdp::sdp_params const& params,
    c10::OptionalRef<sdp::SDPDiagnostics> diagnostics) {
  const bool any_inputs_require_grad = params.query.requires_grad() ||
      params.key.requires_grad() || params.value.requires_grad();
  const bool gradmode_enabled = at::GradMode::is_enabled();
  if (any_inputs_require_grad && gradmode_enabled) {
    sdp::report_failure(diagnostics, "Backward or grad to be supported.");
  }
  return !any_inputs_require_grad || !gradmode_enabled;
}

bool check_no_grad(sdp::sdp_params const& params, bool debug) {
  if (!debug) {
    return check_no_grad(params, c10::OptionalRef<sdp::SDPDiagnostics>{});
  }
  sdp::SDPDiagnostics diagnostics(true);
  return check_no_grad(params, diagnostics);
}

bool can_use_overrideable_attention(
    sdp::sdp_params const& params,
    c10::OptionalRef<sdp::SDPDiagnostics> diagnostics) {
  constexpr auto supported_dtypes = c10::array_of<at::ScalarType>(
      at::kFloat, at::kBFloat16, at::kHalf); // double is not supported

  constexpr auto constraints = c10::array_of<bool (*)(
      sdp::sdp_params const&, c10::OptionalRef<sdp::SDPDiagnostics>)>(
      sdp::check_nested_tensor,
      sdp::check_for_dropout,
      sdp::check_tensor_shapes,
      sdp::check_batch_size_and_num_heads_dense<true /*supports GQA*/>,
      sdp::check_attn_mask_shape,
      sdp::check_nonzero_sequence_lengths_dense,
      sdp::check_last_dim_stride_equals_1_dense<false /*ignore_singleton_dim*/>,
      check_head_dim_size_xpu,
      check_no_grad);
  for (const auto& constraint : constraints) {
    if (!constraint(params, diagnostics)) {
      return false;
    }
  }
  return sdp::check_tensor_dtype(params, supported_dtypes, diagnostics);
}

bool can_use_overrideable_attention(sdp::sdp_params const& params, bool debug) {
  if (!debug) {
    return can_use_overrideable_attention(
        params, c10::OptionalRef<sdp::SDPDiagnostics>{});
  }
  sdp::SDPDiagnostics diagnostics(true);
  return can_use_overrideable_attention(params, diagnostics);
}

bool can_use_cudnn_attention(
    sdp::sdp_params const& params,
    c10::OptionalRef<sdp::SDPDiagnostics> diagnostics) {
  sdp::report_failure(
      diagnostics, "XPU don't support SDPA cudnn attention backend.");
  return false;
}

bool can_use_cudnn_attention(sdp::sdp_params const& params, bool debug) {
  if (!debug) {
    return can_use_cudnn_attention(
        params, c10::OptionalRef<sdp::SDPDiagnostics>{});
  }
  sdp::SDPDiagnostics diagnostics(true);
  return can_use_cudnn_attention(params, diagnostics);
}

int64_t minimum_gemm_alignment(sdp::sdp_params const& params) {
  bool is_half = (params.query.dtype() == at::kHalf) ||
      (params.query.dtype() == at::kBFloat16);
  int64_t matmul_alignment_mn = 4;
  int64_t bits_per_scalar = is_half ? 16 : 32;
  matmul_alignment_mn = std::max(matmul_alignment_mn, 128 / bits_per_scalar);

  return matmul_alignment_mn;
}

bool check_head_dim_size_mem_efficient(
    sdp::sdp_params const& params,
    c10::OptionalRef<sdp::SDPDiagnostics> diagnostics) {
  const auto query_size_last = params.query.sym_size(-1);
  const auto value_size_last = params.value.sym_size(-1);
  const int64_t alignment = minimum_gemm_alignment(params);
  if (!(query_size_last == params.key.sym_size(-1) &&
        query_size_last % alignment == 0 && query_size_last > 0 &&
        value_size_last % alignment == 0 && value_size_last > 0)) {
    sdp::report_failure(
        diagnostics,
        "Mem efficient attention requires last dimension of inputs to be divisible by ",
        alignment,
        ". ",
        "Got Query.size(-1): ",
        query_size_last,
        ", Key.size(-1): ",
        params.key.sym_size(-1),
        ", Value.size(-1): ",
        params.value.sym_size(-1),
        " instead.");
    return false;
  }
  return true;
}

bool check_head_dim_size_mem_efficient(
    sdp::sdp_params const& params,
    bool debug) {
  if (!debug) {
    return check_head_dim_size_mem_efficient(
        params, c10::OptionalRef<sdp::SDPDiagnostics>{});
  }
  sdp::SDPDiagnostics diagnostics(true);
  return check_head_dim_size_mem_efficient(params, diagnostics);
}

bool can_use_mem_efficient_attention(
    sdp::sdp_params const& params,
    c10::OptionalRef<sdp::SDPDiagnostics> diagnostics) {
  constexpr auto general_constraints = c10::array_of<bool (*)(
      sdp::sdp_params const&, c10::OptionalRef<sdp::SDPDiagnostics>)>(
      sdp::check_runtime_disabled_mem_efficient,
      sdp::check_tensor_shapes,
      check_head_dim_size_mem_efficient);
  for (const auto& constraint : general_constraints) {
    if (!constraint(params, diagnostics)) {
      return false;
    }
  }
  if (has_for_nested_inputs(params)) {
    constexpr auto nested_constraints = c10::array_of<bool (*)(
        sdp::sdp_params const&, c10::OptionalRef<sdp::SDPDiagnostics>)>(
        sdp::check_requires_grad_and_nested,
        sdp::check_batch_size_nested,
        sdp::check_for_seq_len_0_nested_tensor);
    for (const auto& constraint : nested_constraints) {
      if (!constraint(params, diagnostics)) {
        return false;
      }
    }
  }
  if (has_only_dense_inputs(params)) {
    constexpr auto dense_constraints = c10::array_of<bool (*)(
        sdp::sdp_params const&, c10::OptionalRef<sdp::SDPDiagnostics>)>(
        sdp::check_nonzero_sequence_lengths_dense,
        sdp::check_last_dim_stride_equals_1_dense<false>,
        sdp::check_batch_size_and_num_heads_dense<false>);
    for (const auto& constraint : dense_constraints) {
      if (!constraint(params, diagnostics)) {
        return false;
      }
    }
  }
  return true;
}

bool can_use_mem_efficient_attention(
    sdp::sdp_params const& params,
    bool debug) {
  if (!debug) {
    return can_use_mem_efficient_attention(
        params, c10::OptionalRef<sdp::SDPDiagnostics>{});
  }
  sdp::SDPDiagnostics diagnostics(true);
  return can_use_mem_efficient_attention(params, diagnostics);
}

bool priority_order_init = false;

std::array<sdp::SDPBackend, sdp::num_backends> priority_order(
    sdp::sdp_params const& params) {
  if (!priority_order_init) {
    priority_order_init = true;
    const std::vector<int64_t> priority_order = {
        static_cast<int64_t>(at::SDPBackend::overrideable),
        static_cast<int64_t>(at::SDPBackend::flash_attention),
        static_cast<int64_t>(at::SDPBackend::math),
        static_cast<int64_t>(at::SDPBackend::efficient_attention),
        static_cast<int64_t>(at::SDPBackend::cudnn_attention)};
    at::globalContext().setSDPPriorityOrder(priority_order);
  }
  return at::globalContext().sDPPriorityOrder();
}

sdp::SDPBackend select_sdp_backend_xpu(sdp::sdp_params const& kernel_params) {
  // This function defines the priority order of the different sdp backends
  // 1. Flash Attention
  // 2. Math fallback
  auto& ctx = at::globalContext();

  // Get ideal kernel ordering
  const auto ordering = priority_order(kernel_params);

  // Because TORCHCHECK checks if condition is true we negate debug so that
  // The statements will be printed when debug is true
  for (const auto& backend : ordering) {
    switch (backend) {
      case sdp::SDPBackend::overrideable:
        if (ctx.userEnabledOverrideableSDP() &&
            can_use_overrideable_attention(kernel_params, false)) {
          return sdp::SDPBackend::overrideable;
        }
        break;
      case sdp::SDPBackend::math:
        if (ctx.userEnabledMathSDP()) {
          return sdp::SDPBackend::math;
        }
        break;
      case sdp::SDPBackend::flash_attention:
        if (ctx.userEnabledFlashSDP() &&
            sdp::can_use_flash_attention(kernel_params, false)) {
          return sdp::SDPBackend::flash_attention;
        }
        break;
      case sdp::SDPBackend::cudnn_attention:
        if (ctx.userEnabledCuDNNSDP() &&
            can_use_cudnn_attention(kernel_params, false)) {
          TORCH_CHECK(false, "Invalid backend");
        }
        break;
      case sdp::SDPBackend::efficient_attention:
        if (ctx.userEnabledMemEfficientSDP() &&
            can_use_mem_efficient_attention(kernel_params, false)) {
          TORCH_WARN_ONCE(
              "SDPA Memory Efficient Attention backend is not supported on XPU, falling back to math backend.");
          return sdp::SDPBackend::math;
        }
        break;
      default:
        TORCH_CHECK(false, "Invalid backend");
    }
  }
  sdp::SDPDiagnostics diagnostics(false);
  diagnostics.set_current_backend("Flash attention");
  if (ctx.userEnabledFlashSDP()) {
    sdp::can_use_flash_attention(kernel_params, diagnostics);
  } else {
    sdp::report_failure(
        diagnostics, "Flash attention has been runtime disabled.");
  }
  diagnostics.set_current_backend("OneDNN attention");
  if (ctx.userEnabledOverrideableSDP()) {
    can_use_overrideable_attention(kernel_params, diagnostics);
  } else {
    sdp::report_failure(
        diagnostics, "OneDNN attention has been runtime disabled.");
  }
  diagnostics.set_current_backend("Math attention");
  if (!ctx.userEnabledMathSDP()) {
    sdp::report_failure(
        diagnostics, "Math attention has been runtime disabled.");
  }
  diagnostics.set_current_backend("cuDNN attention");
  if (ctx.userEnabledCuDNNSDP()) {
    can_use_cudnn_attention(kernel_params, diagnostics);
  } else {
    sdp::report_failure(
        diagnostics, "cuDNN attention has been runtime disabled.");
  }
  diagnostics.set_current_backend("Memory efficient attention");
  can_use_mem_efficient_attention(kernel_params, diagnostics);
  diagnostics.raise_error();
  return sdp::SDPBackend::error;
}
} // namespace

namespace at::native {
int64_t _fused_sdp_choice_xpu(
    const at::Tensor& query_,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask_,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa) {
  sdp::sdp_params kernel_params{
      query_, key, value, attn_mask_, dropout_p, is_causal, enable_gqa};
  auto backend = select_sdp_backend_xpu(kernel_params);

  if (backend == sdp::SDPBackend::error) {
    TORCH_CHECK(
        false,
        "No viable backend for scaled_dot_product_attention was found. ",
        "This is likely due to turning off both the math kernel and the overrideable kernels.");
  }
  return static_cast<int64_t>(backend);
}

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    c10::SymInt,
    c10::SymInt,
    at::Tensor,
    at::Tensor,
    at::Tensor>
_scaled_dot_product_fused_attention_overrideable_xpu(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_bias,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale) {
  TORCH_INTERNAL_ASSERT(
      query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
      "scaled_dot_product_fused_attention_overrideable_xpu: Accept only 4 dims inputs shape of {(B), H, T, K}");
  TORCH_INTERNAL_ASSERT(
      (key.size(0) == value.size(0)) && (key.size(1) == value.size(1)) &&
          (key.size(2) == value.size(2)),
      "scaled_dot_product_fused_attention_overrideable_xpu: K/V should have the same batch / seq / num_head");
  TORCH_INTERNAL_ASSERT(
      query.size(3) == key.size(3),
      "scaled_dot_product_fused_attention_overrideable_xpu: Q/K should have the same head_dim");
  TORCH_INTERNAL_ASSERT(
      query.size(1) % key.size(1) == 0,
      "scaled_dot_product_fused_attention_overrideable_xpu: number of heads in K/V must divide number of heads in Q");
  TORCH_INTERNAL_ASSERT(
      dropout_p == 0.0,
      "scaled_dot_product_fused_attention_overrideable_xpu: Currently do not support dropout > 0");
  TORCH_INTERNAL_ASSERT(
      !(attn_bias.has_value() && is_causal),
      "scaled_dot_product_fused_attention_overrideable_xpu: attn_bias cannot present with is_causal");

  const int64_t batch_size = query.size(0);
  const int64_t num_head_q = query.size(1);
  const int64_t num_head_kv = key.size(1);
  const int64_t head_dim_qk = query.size(3);
  const int64_t head_dim_v = value.size(3);
  const int64_t seq_len_q = query.size(2);
  const int64_t seq_len_kv = key.size(2);

  at::Tensor output;
  std::vector<int64_t> output_shape = {
      batch_size, num_head_q, seq_len_q, head_dim_v};
  alloc_with_matching_layout(query, output, output_shape);
  at::Tensor logsumexp, debug_attn_mask; // not supported

  at::native::onednn::sdpa(
      batch_size,
      seq_len_q,
      seq_len_kv,
      num_head_q,
      num_head_kv,
      head_dim_qk,
      head_dim_v,
      query,
      key,
      value,
      attn_bias,
      is_causal,
      scale.has_value() ? scale.value() : (1.0 / std::sqrt(head_dim_qk)),
      output,
      false,
      logsumexp);

  // rng not used
  auto philox_seed = at::empty({}, at::dtype(at::kLong));
  auto philox_offset = at::empty({}, at::dtype(at::kLong));
  return std::make_tuple(
      output,
      logsumexp,
      /* cum_seq_q */ at::Tensor(),
      /* cum_seq_k */ at::Tensor(),
      seq_len_q,
      seq_len_kv,
      philox_seed,
      philox_offset,
      debug_attn_mask);
}

REGISTER_XPU_DISPATCH(_fused_sdp_choice_stub, &_fused_sdp_choice_xpu);
} // namespace at::native
