#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <ATen/native/transformers/xpu/flash_attn/flash_api.h>
#include <ATen/ops/constant_pad_nd.h>
#include <ATen/ops/dropout.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/triu.h>

#include <limits>

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

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> _flash_attention_forward_xpu(
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
  TORCH_CHECK(!cum_seq_q.has_value(), "_flash_attention_forward_xpu: cum_seq_q is not supported");
  TORCH_CHECK(!cum_seq_k.has_value(), "_flash_attention_forward_xpu: cum_seq_k is not supported");
    TORCH_CHECK(max_q == query.size(1), "_flash_attention_forward_xpu: max_q must match query.size(1)");
    TORCH_CHECK(max_k == key.size(1), "_flash_attention_forward_xpu: max_k must match key.size(1)");
    TORCH_CHECK(dropout_p >= 0.0 && dropout_p < 1.0, "_flash_attention_forward_xpu: dropout_p must be in [0, 1)");
  TORCH_CHECK(
      !window_size_left.has_value() && !window_size_right.has_value(),
      "_flash_attention_forward_xpu: sliding window attention is not supported");
  TORCH_CHECK(
      !seqused_k.has_value() && !alibi_slopes.has_value() && !block_table.has_value() &&
          !num_splits.has_value(),
      "_flash_attention_forward_xpu: seqused_k/alibi_slopes/block_table/num_splits are not supported");

    const double effective_scale =
            scale.value_or(1.0 / std::sqrt(static_cast<double>(query.size(-1))));

    // Fallback for dropout>0 because the current sycltla flash kernel path is no-dropout only.
    if (dropout_p > 0.0) {
        Tensor q_t = query.transpose(1, 2);
        Tensor k_t = key.transpose(1, 2);
        Tensor v_t = value.transpose(1, 2);

        Tensor scores = at::matmul(q_t, k_t.transpose(-2, -1));
        scores = scores * effective_scale;
        Tensor scores_fp32 = scores.to(at::kFloat);

        if (is_causal) {
            Tensor causal_mask = at::triu(
                    at::ones({query.size(1), key.size(1)}, query.options().dtype(at::kBool)),
                    1);
            scores_fp32 = scores_fp32.masked_fill(causal_mask, -std::numeric_limits<float>::infinity());
        }

        Tensor logsumexp = scores_fp32.logsumexp(-1);
        Tensor attn = scores_fp32.softmax(-1).to(query.scalar_type());
        attn = at::dropout(attn, dropout_p, true);
        Tensor output = at::matmul(attn, v_t).transpose(1, 2);

        Tensor rng_state = at::zeros({2}, query.options().dtype(kUInt64));
        Tensor unused = at::zeros({}, query.options().dtype(kUInt64));
        Tensor debug_attn_mask = at::empty({0}, query.options());
        return std::make_tuple(output, logsumexp, rng_state, unused, debug_attn_mask);
    }

    const int64_t original_head_dim = query.size(-1);
    int64_t padded_head_dim = original_head_dim;
    if (original_head_dim <= 64) {
        padded_head_dim = 64;
    } else if (original_head_dim <= 96) {
        padded_head_dim = 96;
    } else if (original_head_dim <= 128) {
        padded_head_dim = 128;
    } else if (original_head_dim <= 192) {
        padded_head_dim = 192;
    }
    TORCH_CHECK(
            padded_head_dim == 64 || padded_head_dim == 96 || padded_head_dim == 128 ||
                    padded_head_dim == 192,
            "_flash_attention_forward_xpu: unsupported head_dim=",
            original_head_dim);

    const int64_t pad_amount = padded_head_dim - original_head_dim;
    Tensor query_for_kernel = query;
    Tensor key_for_kernel = key;
    Tensor value_for_kernel = value;
    if (pad_amount > 0) {
        query_for_kernel = at::constant_pad_nd(query_for_kernel, {0, pad_amount}, 0);
        key_for_kernel = at::constant_pad_nd(key_for_kernel, {0, pad_amount}, 0);
        value_for_kernel = at::constant_pad_nd(value_for_kernel, {0, pad_amount}, 0);
    }

    Tensor q_t = query_for_kernel.transpose(1, 2);
    Tensor k_t = key_for_kernel.transpose(1, 2);
    Tensor v_t = value_for_kernel.transpose(1, 2);

  auto [attention, logsumexp, _, __, ___, ____, philox_seed, philox_offset, debug_attn_mask] =
      _scaled_dot_product_flash_attention_xpu(
          q_t,
          k_t,
          v_t,
          dropout_p,
          is_causal,
          return_debug_mask,
                    effective_scale);

  Tensor output = attention.transpose(1, 2);

    if (logsumexp.scalar_type() != kFloat) {
        logsumexp = logsumexp.to(kFloat);
    }

    Tensor philox_seed_uint64 = philox_seed;
    if (philox_seed_uint64.scalar_type() != kUInt64) {
        philox_seed_uint64 = philox_seed_uint64.to(kUInt64);
    }
    if (philox_seed_uint64.dim() == 0) {
        philox_seed_uint64 = philox_seed_uint64.unsqueeze(0).expand({2});
    }

    Tensor philox_offset_uint64 = philox_offset;
    if (philox_offset_uint64.scalar_type() != kUInt64) {
        philox_offset_uint64 = philox_offset_uint64.to(kUInt64);
    }
    if (philox_offset_uint64.dim() > 0) {
        philox_offset_uint64 = philox_offset_uint64.squeeze();
    }

    if (pad_amount > 0) {
        output = output.narrow(-1, 0, original_head_dim);
    }
    return std::make_tuple(
            output,
            logsumexp,
            philox_seed_uint64,
            philox_offset_uint64,
            debug_attn_mask);
}

} // namespace native
} // namespace at
