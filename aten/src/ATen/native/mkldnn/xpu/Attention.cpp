#include <ATen/Context.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNN.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <c10/util/Array.h>
#include <torch/library.h>

namespace {
bool check_head_dim_size_xpu(sdp::sdp_params const& params, bool debug) {
  const auto query_size_last = params.query.sym_size(-1);
  const auto key_size_last = params.key.sym_size(-1);
  const auto value_size_last = params.value.sym_size(-1);
  if (query_size_last != key_size_last) {
    if (debug) {
      TORCH_WARN(
          "OneDNN attention requires q,k to have the same last dimension.",
          " Got Query.size(-1): ",
          query_size_last,
          ", Key.size(-1): ",
          key_size_last,
          " instead.");
    }
    return false;
  }

  constexpr int MAX_HEAD_DIM = 576;
  const auto max_size_last = query_size_last.max(value_size_last);
  if (max_size_last > MAX_HEAD_DIM) {
    if (debug) {
      TORCH_WARN(
          "OneDNN attention requires q,k,v to have head dimension less than ",
          MAX_HEAD_DIM,
          ". Got ",
          max_size_last,
          " instead.");
    }
    return false;
  }
  return true;
}

bool input_require_grad(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask) {
  return at::GradMode::is_enabled() &&
      (query.requires_grad() || key.requires_grad() || value.requires_grad() ||
       (attn_mask.has_value() && attn_mask.value().requires_grad()));
}

bool check_grad(sdp::sdp_params const& params, bool debug) {
  if (!input_require_grad(
          params.query, params.key, params.value, params.attn_mask))
    return true;

  auto q_num_heads = params.query.sym_size(-3);
  auto k_num_heads = params.key.sym_size(-3);
  auto v_num_heads = params.value.sym_size(-3);
  bool is_gqa = q_num_heads != k_num_heads || q_num_heads != v_num_heads;
  if (debug && is_gqa)
    TORCH_WARN(
        "scale_dot_product_attention with gqa is not supported for gradient computation on xpu.");

  bool attn_mask_needs_grad =
      params.attn_mask.has_value() && params.attn_mask.value().requires_grad();
  if (debug && attn_mask_needs_grad) {
    TORCH_WARN(
        "scale_dot_product_attention on xpu is not supported when attn_mask.requires_grad() == True.");
  }

  return !is_gqa && !attn_mask_needs_grad;
}

bool can_use_overrideable_attention(sdp::sdp_params const& params, bool debug) {
  constexpr auto supported_dtypes = c10::array_of<at::ScalarType>(
      at::kFloat, at::kBFloat16, at::kHalf); // double is not supported

  // Define gate functions that determine if a flash kernel can be run
  constexpr auto constraints = c10::array_of<bool (*)(
      sdp::sdp_params const&, bool)>(
      sdp::check_nested_tensor,
      sdp::check_for_dropout,
      sdp::check_tensor_shapes,
      sdp::check_batch_size_and_num_heads_dense<true /*supports GQA*/>,
      sdp::check_attn_mask_shape,
      sdp::check_nonzero_sequence_lengths_dense,
      sdp::check_last_dim_stride_equals_1_dense<false /*ignore_singleton_dim*/>,
      check_head_dim_size_xpu,
      check_grad);
  for (auto& constraint : constraints) {
    if (!constraint(params, debug)) {
      return false;
    }
  }
  return sdp::check_tensor_dtype(params, supported_dtypes, debug);
}

bool can_use_flash_attention(sdp::sdp_params const& params, bool debug) {
  // Currently, XPU fallbacks flash attention to overridable
  return can_use_overrideable_attention(params, debug);
}

bool can_use_cudnn_attention(sdp::sdp_params const& params, bool debug) {
  if (debug) {
    TORCH_WARN("XPU don't support SDPA cudnn attention backend.");
  }
  return false;
}

bool can_use_mem_efficien_attention(sdp::sdp_params const& params, bool debug) {
  if (debug) {
    TORCH_WARN("XPU don't support SDPA mem efficient attention backend.");
  }
  return false;
}

bool priority_order_init = false;

std::array<sdp::SDPBackend, sdp::num_backends> priority_order(
    sdp::sdp_params const& params) {
  if (!priority_order_init) {
    priority_order_init = true;
    const std::vector<int64_t> priority_order = {
        static_cast<int64_t>(at::SDPBackend::overrideable),
        static_cast<int64_t>(at::SDPBackend::math),
        static_cast<int64_t>(at::SDPBackend::flash_attention),
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
  // use overridable linked to onednn as overridable implementation
  if (!ctx.userEnabledMathSDP() && !ctx.userEnabledOverrideableSDP() &&
      !ctx.userEnabledFlashSDP()) {
    return sdp::SDPBackend::error;
  }

  // Get ideal kernel ordering
  const auto ordering = priority_order(kernel_params);

  // Because TORCHCHECK checks if condition is true we negate debug so that
  // The statements will be printed when debug is true
  bool print_debug = false;
  for (auto& backend : ordering) {
    switch (backend) {
      case sdp::SDPBackend::overrideable:
        if (ctx.userEnabledOverrideableSDP() &&
            can_use_overrideable_attention(kernel_params, print_debug)) {
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
            can_use_flash_attention(kernel_params, print_debug)) {
          TORCH_WARN_ONCE(
              "SDPA Flash Attention backend is not supported on XPU, falling back to OVERRIDEABLE backend.");
          return sdp::SDPBackend::overrideable;
        }
        break;
      case sdp::SDPBackend::cudnn_attention:
        if (ctx.userEnabledCuDNNSDP() &&
            can_use_cudnn_attention(kernel_params, print_debug)) {
          TORCH_CHECK(false, "Invalid backend");
        }
        break;
      case sdp::SDPBackend::efficient_attention:
        if (ctx.userEnabledMemEfficientSDP() &&
            can_use_mem_efficien_attention(kernel_params, print_debug)) {
          TORCH_CHECK(false, "Invalid backend");
        }
        break;
      default:
        TORCH_CHECK(false, "Invalid backend");
    }
  }
  // If we have gotten to this point then two things have happened:
  // 1. can_use_overridable_attention did not satisfy the constraints to be ran
  // 2. The user has explicitly disabled the math kernel
  // We then re-run the kernel checks with debug enabled to print out the
  // reason why the kernel was not selected

  print_debug = true;
  TORCH_WARN("Flash attention kernel not used because:");
  can_use_flash_attention(kernel_params, print_debug);
  TORCH_WARN("Overrideable attention kernel not used because:");
  can_use_overrideable_attention(kernel_params, print_debug);
  TORCH_WARN("CuDNN attention kernel not used because:");
  can_use_cudnn_attention(kernel_params, print_debug);
  TORCH_WARN("Memory Efficient attention kernel not used because:");
  can_use_mem_efficien_attention(kernel_params, print_debug);
  TORCH_CHECK(!print_debug, "No available kernel. Aborting execution.")
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
    std::optional<double> scale,
    bool compute_logsumexp) {
  TORCH_INTERNAL_ASSERT(
      query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
      "scaled_dot_product_fused_attention_overrideable_xpu: Accept only 4 dims inputs shape of {B, H, T, K}");
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
  TORCH_INTERNAL_ASSERT(
      !(attn_bias.has_value() && attn_bias.value().requires_grad()),
      "scaled_dot_product_fused_attention_overrideable_xpu: attn_bias cannot have requires_grad=True");

  const int64_t batch_size = query.size(0);
  const int64_t num_head_q = query.size(1);
  const int64_t num_head_kv = key.size(1);
  const int64_t head_dim_qk = query.size(3);
  const int64_t head_dim_v = value.size(3);
  const int64_t seq_len_q = query.size(2);
  const int64_t seq_len_kv = key.size(2);

  at::Tensor attention;
  std::vector<int64_t> attention_shape = {
      batch_size, num_head_q, seq_len_q, head_dim_v};
  alloc_with_matching_layout(query, attention, attention_shape);

  auto opts = query.options();
  at::Tensor logsumexp =
      at::empty({batch_size, num_head_q, seq_len_q}, opts.dtype(at::kFloat));

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
      attention,
      compute_logsumexp,
      logsumexp);

  // rng not used
  auto philox_seed = at::empty({}, at::dtype(at::kLong));
  auto philox_offset = at::empty({}, at::dtype(at::kLong));
  return std::make_tuple(
      attention,
      logsumexp,
      /* cum_seq_q */ at::Tensor(),
      /* cum_seq_k */ at::Tensor(),
      seq_len_q,
      seq_len_kv,
      philox_seed,
      philox_offset,
      /*debug_attn_mask */ at::Tensor());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_scaled_dot_product_fused_attention_overrideable_backward_xpu(
    const at::Tensor& grad_out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& attn_bias,
    std::array<bool, 4> grad_input_mask,
    const at::Tensor& out,
    const at::Tensor& logsumexp,
    const at::Tensor& cum_seq_q,
    const at::Tensor& cum_seq_k,
    int64_t max_q,
    int64_t max_k,
    double dropout_p,
    bool is_causal,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    std::optional<double> scale) {
  TORCH_INTERNAL_ASSERT(
      grad_out.dim() == 4 && out.dim() == 4 &&
          grad_out.size(0) == out.size(0) && grad_out.size(1) == out.size(1) &&
          grad_out.size(2) == out.size(2) && grad_out.size(3) == out.size(3),
      "scaled_dot_product_fused_attention_overrideable_backward_xpu: grad_out and out should have the same shape of {B, H, T, K}");
  TORCH_INTERNAL_ASSERT(
      query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
      "scaled_dot_product_fused_attention_overrideable_backward_xpu: Accept only 4 dims inputs shape of {B, H, T, K}");
  TORCH_INTERNAL_ASSERT(
      (key.size(0) == value.size(0)) && (key.size(1) == value.size(1)) &&
          (key.size(2) == value.size(2)),
      "scaled_dot_product_fused_attention_overrideable_backward_xpu: K/V should have the same batch / seq / num_head");
  TORCH_INTERNAL_ASSERT(
      query.size(0) == grad_out.size(0) && query.size(1) == grad_out.size(1) &&
          query.size(2) == grad_out.size(2),
      "scaled_dot_product_fused_attention_overrideable_backward_xpu: Q should have the same batch / num_head / seq_len as grad_out");
  TORCH_INTERNAL_ASSERT(
      query.size(3) == key.size(3),
      "scaled_dot_product_fused_attention_overrideable_backward_xpu: Q/K should have the same head_dim");
  TORCH_INTERNAL_ASSERT(
      value.size(3) == grad_out.size(3),
      "scaled_dot_product_fused_attention_overrideable_backward_xpu: V should have the same head_dim as grad_out");
  TORCH_INTERNAL_ASSERT(
      query.size(1) == key.size(1),
      "scaled_dot_product_fused_attention_overrideable_backward_xpu: number of heads in K/V must equal to number of heads in Q");
  TORCH_INTERNAL_ASSERT(
      dropout_p == 0.0,
      "scaled_dot_product_fused_attention_overrideable_backward_xpu: Currently do not support dropout > 0");
  TORCH_INTERNAL_ASSERT(
      logsumexp.dim() == 3 && logsumexp.size(0) == query.size(0) &&
      logsumexp.size(1) == query.size(1) &&
      logsumexp.size(2) == query.size(2) &&
      "scaled_dot_product_fused_attention_overrideable_backward_xpu: logsumexp should have the shape of {B, H, T}");

  std::optional<Tensor> attn_bias_opt;
  if (attn_bias.defined()) {
    attn_bias_opt = attn_bias;
  }

  const int64_t batch_size = query.size(0);
  const int64_t num_head_q = query.size(1);
  const int64_t num_head_kv = key.size(1);
  const int64_t seq_len_q = query.size(2);
  const int64_t seq_len_kv = key.size(2);
  const int64_t head_dim_qk = query.size(3);
  const int64_t head_dim_v = value.size(3);

  auto grad_q = at::empty_like(query);
  auto grad_k = at::empty_like(key);
  auto grad_v = at::empty_like(value);
  auto grad_attn_bias = attn_bias_opt.has_value()
      ? at::empty_like(attn_bias_opt.value())
      : at::Tensor();
  at::native::onednn::sdpa_backward(
      batch_size,
      num_head_q,
      num_head_kv,
      seq_len_q,
      seq_len_kv,
      head_dim_qk,
      head_dim_v,
      grad_out,
      query,
      key,
      value,
      out,
      logsumexp,
      attn_bias_opt,
      is_causal,
      scale.has_value() ? scale.value() : (1.0 / std::sqrt(query.size(3))),
      grad_q,
      grad_k,
      grad_v);
  return std::make_tuple(
      std::move(grad_q),
      std::move(grad_k),
      std::move(grad_v),
      std::move(grad_attn_bias));
}

REGISTER_XPU_DISPATCH(_fused_sdp_choice_stub, &_fused_sdp_choice_xpu);
} // namespace at::native
