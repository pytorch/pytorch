#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <ATen/native/transformers/xpu/flash_attn/flash_api.h>

namespace at {
namespace native {

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> _flash_attention_forward_xpu(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& cumulative_sequence_length_q,
    const std::optional<Tensor>& cumulative_sequence_length_k,
    int64_t max_seqlen_batch_q,
    int64_t max_seqlen_batch_k,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale,
    std::optional<int64_t> window_size_left,
    std::optional<int64_t> window_size_right,
    const std::optional<Tensor>& _seqused_k,
    const std::optional<Tensor>& _alibi_slopes,
    const std::optional<Tensor>& _block_table,
    std::optional<int64_t> num_splits) {
  return sycltla::_flash_attention_forward(
      query,
      key,
      value,
      cumulative_sequence_length_q,
      cumulative_sequence_length_k,
      max_seqlen_batch_q,
      max_seqlen_batch_k,
      dropout_p,
      is_causal,
      return_debug_mask,
      scale,
      window_size_left,
      window_size_right,
      _seqused_k,
      _alibi_slopes,
      _block_table,
      /*out=*/std::nullopt,
      num_splits);
}

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
  // Used for tracking usage statistics
  C10_LOG_API_USAGE_ONCE("torch.sdpa.flash_attention");
  // Query (Batch x Num_heads x Q_seq_len  x Dim_per_head)
  // Key   (Batch x Num_heads x KV_seq_len x Dim_per_head)
  // Value (Batch x Num_heads x KV_seq_len x Dim_per_head)

  const int64_t max_seqlen_batch_q = query.size(2);
  const int64_t max_seqlen_batch_k = key.size(2);
  const int64_t max_seqlen_batch_v = value.size(2);
  TORCH_CHECK(
      max_seqlen_batch_k == max_seqlen_batch_v,
      "Key and Value must have the same sequence length");

  // Query -> Query(Batch x Q_seq_len  x Num_heads x Dim_per_head)
  // Key   -> Key  (Batch x KV_seq_len x Num_heads x Dim_per_head)
  // Value -> Value(Batch x KV_seq_len x Num_heads x Dim_per_head)
  Tensor q_t = query.transpose(1, 2);
  Tensor k_t = key.transpose(1, 2);
  Tensor v_t = value.transpose(1, 2);

  auto [output, logsumexp, philox_seed, philox_offset, debug_attn_mask] =
      at::_flash_attention_forward(
          q_t,
          k_t,
          v_t,
          std::nullopt,
          std::nullopt,
          max_seqlen_batch_q,
          max_seqlen_batch_k,
          dropout_p,
          is_causal,
          return_debug_mask,
          scale,
          std::nullopt,
          std::nullopt);
  // Reshape output to convert nnz to batch_size and seq_len
  Tensor attention = output.transpose(1, 2);

  return std::make_tuple(
      std::move(attention),
      std::move(logsumexp),
      Tensor(),
      Tensor(),
      max_seqlen_batch_q,
      max_seqlen_batch_k,
      std::move(philox_seed),
      std::move(philox_offset),
      std::move(debug_attn_mask));
}

} // namespace native
} // namespace at
