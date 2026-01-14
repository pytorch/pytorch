#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <ATen/native/transformers/xpu/flash_attn/flash_api.h>

namespace at {
namespace native {

std::tuple<at::Tensor, at::Tensor, at::Tensor>
_scaled_dot_product_flash_attention_backward_xpu(
    const at::Tensor& grad_out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const at::Tensor& logsumexp,
    const at::Tensor& cumulative_sequence_length_q,
    const at::Tensor& cumulative_sequence_length_k,
    const int64_t max_seqlen_batch_q,
    const int64_t max_seqlen_batch_k,
    double dropout_p,
    bool is_causal,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    std::optional<double> scale) {
  if (!grad_out.defined()) {
    return std::make_tuple(Tensor{}, Tensor{}, Tensor{});
  }

  auto [grad_q, grad_k, grad_v] = sycltla::flash_attention_backward(
      grad_out,
      query,
      key,
      value,
      out,
      logsumexp,
      cumulative_sequence_length_q,
      cumulative_sequence_length_k,
      max_seqlen_batch_q,
      max_seqlen_batch_k,
      dropout_p,
      is_causal,
      philox_seed,
      philox_offset,
      scale.has_value() ? scale.value() : (1.0 / std::sqrt(query.size(3))));

  return std::make_tuple(
      std::move(grad_q), std::move(grad_k), std::move(grad_v));
}

} // namespace native
} // namespace at
