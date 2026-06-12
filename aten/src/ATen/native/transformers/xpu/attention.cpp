#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <ATen/native/transformers/xpu/flash_attn/flash_api.h>

namespace at {
namespace native {

std::tuple<
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    c10::SymInt,
    c10::SymInt,
    Tensor,
    Tensor,
    Tensor>
_scaled_dot_product_flash_attention_xpu(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale) {
  auto
      [attention,
       logsumexp,
       cumulative_sequence_length_q,
       cumulative_sequence_length_k,
       max_seqlen_batch_q,
       max_seqlen_batch_k,
       philox_seed,
       philox_offset] =
          sycltla::flash_attention_forward(
              query,
              key,
              value,
              dropout_p,
              is_causal,
              scale.has_value() ? scale.value()
                                : (1.0 / std::sqrt(query.size(3))));
  return std::make_tuple(
      attention,
      logsumexp,
      cumulative_sequence_length_q,
      cumulative_sequence_length_k,
      max_seqlen_batch_q,
      max_seqlen_batch_k,
      philox_seed,
      philox_offset,
      /* debug_attn_mask */ at::Tensor());
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
_flash_attention_forward_xpu(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& cum_seq_q,
    const std::optional<Tensor>& cum_seq_k,
    int64_t max_q,
    int64_t max_k,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale,
    std::optional<int64_t> window_size_left,
    std::optional<int64_t> window_size_right,
    const std::optional<Tensor>& seqused_k,
    const std::optional<Tensor>& alibi_slopes,
    const std::optional<Tensor>& block_table,
    std::optional<int64_t> num_splits) {
  TORCH_CHECK(
      !window_size_left.has_value() && !window_size_right.has_value(),
      "_flash_attention_forward: window_size_left and window_size_right are not supported on XPU");
  TORCH_CHECK(
      !seqused_k.has_value(),
      "_flash_attention_forward: seqused_k is not supported on XPU");
  TORCH_CHECK(
      !alibi_slopes.has_value(),
      "_flash_attention_forward: alibi_slopes is not supported on XPU");
  TORCH_CHECK(
      !block_table.has_value(),
      "_flash_attention_forward: block_table (paged attention) is not supported on XPU");
  TORCH_CHECK(
      !num_splits.has_value(),
      "_flash_attention_forward: num_splits is not supported on XPU");
  TORCH_CHECK(
      dropout_p == 0.0,
      "_flash_attention_forward: dropout is not yet properly supported on XPU (RNG state handling not implemented)");

  // Validate dtype early for better error messages
  auto dtype = query.scalar_type();
  TORCH_CHECK(
      dtype == at::kHalf || dtype == at::kBFloat16,
      "_flash_attention_forward: only fp16 and bf16 data types are supported on XPU, got ",
      dtype);
  TORCH_CHECK(
      key.scalar_type() == dtype && value.scalar_type() == dtype,
      "_flash_attention_forward: query, key, and value must have the same dtype");

  const float scale_val = scale.has_value()
      ? static_cast<float>(scale.value())
      : static_cast<float>(1.0 / std::sqrt(query.size(-1)));

  if (cum_seq_q.has_value() && cum_seq_k.has_value()) {
    // Varlen (packed/nested tensor) path
    // Ensure inputs are contiguous in the last dimension for SYCLTLA backend
    const auto& query_c = query.stride(-1) == 1 ? query : query.contiguous();
    const auto& key_c = key.stride(-1) == 1 ? key : key.contiguous();
    const auto& value_c = value.stride(-1) == 1 ? value : value.contiguous();

    auto [output, logsumexp] = sycltla::flash_attention_forward_varlen(
        query_c,
        key_c,
        value_c,
        cum_seq_q.value(),
        cum_seq_k.value(),
        max_q,
        max_k,
        dropout_p,
        is_causal,
        scale_val);
    // RNG state for dropout replay: not implemented yet, return empty tensor
    at::Tensor rng_state = at::empty({}, at::dtype(at::kLong).device(query.device()));
    return std::make_tuple(
        output,
        logsumexp,
        rng_state,
        /* unused */ at::Tensor(),
        /* debug_attn_mask */ at::Tensor());
  }

  // Dense path: delegate to existing implementation
  // Ensure inputs are contiguous for SYCLTLA backend
  const auto& query_c = query.is_contiguous() ? query : query.contiguous();
  const auto& key_c = key.is_contiguous() ? key : key.contiguous();
  const auto& value_c = value.is_contiguous() ? value : value.contiguous();

  auto [attention, logsumexp, csq, csk, mq, mk, philox_seed, philox_offset] =
      sycltla::flash_attention_forward(
          query_c, key_c, value_c, dropout_p, is_causal, scale_val);
  // Intentionally return an empty tensor as a placeholder for RNG state.
  // This is safe because dropout is not yet properly supported on XPU, as
  // enforced by the TORCH_CHECK above requiring dropout_p == 0.0.
  at::Tensor rng_state = at::empty({}, at::dtype(at::kLong).device(query.device()));
  return std::make_tuple(
      attention,
      logsumexp,
      rng_state,
      /* unused */ at::Tensor(),
      /* debug_attn_mask */ at::Tensor());
}

} // namespace native
} // namespace at
