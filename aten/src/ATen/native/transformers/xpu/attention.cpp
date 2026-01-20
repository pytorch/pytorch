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

} // namespace native
} // namespace at
