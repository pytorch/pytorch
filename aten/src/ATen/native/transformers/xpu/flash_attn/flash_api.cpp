#include <ATen/native/transformers/xpu/flash_attn/flash_api.h>
#include <ATen/native/transformers/xpu/flash_attn/sycltla/flash_api.h>

namespace sycltla {

bool is_flash_attention_available() {
#ifndef USE_SYCLTLA
  return false;
#else
  return true;
#endif
}

std::tuple<at::Tensor, at::Tensor> flash_attention_forward(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const double dropout,
    const bool is_causal,
    const float scale) {
#ifndef USE_SYCLTLA
  TORCH_CHECK(
      false,
      "flash_attention_forward: Torch XPU was not compiled with SYCLTLA support.");
  return std::make_tuple(at::Tensor(), at::Tensor());
#else
  auto [output, attention_weights] = flash_attention_forward_sycltla(
      query,
      key,
      value,
      dropout,
      is_causal,
      scale);
  return std::make_tuple(std::move(output), std::move(attention_weights));
#endif
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> flash_attention_backward(
    const at::Tensor& grad_out,
    const at::Tensor& out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& logsumexp,
    const at::Tensor& cumulative_sequence_length_q,
    const at::Tensor& cumulative_sequence_length_k,
    const int64_t max_seqlen_batch_q,
    const int64_t max_seqlen_batch_k,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    const std::optional<at::Tensor>& attn_mask,
    const double dropout,
    const bool is_causal,
    const float scale) {
#ifndef USE_SYCLTLA
  TORCH_CHECK(
      false,
      "flash_attention_backward: Torch XPU was not compiled with SYCLTLA support.");
  return std::make_tuple(at::Tensor(), at::Tensor(), at::Tensor());
#else
  auto [grad_query, grad_key, grad_value] = flash_attention_backward_sycltla(
      grad_out,
      out,
      query,
      key,
      value,
      logsumexp,
      cumulative_sequence_length_q,
      cumulative_sequence_length_k,
      max_seqlen_batch_q,
      max_seqlen_batch_k,
      philox_seed,
      philox_offset,
      attn_mask,
      dropout,
      is_causal,
      scale);
  return std::make_tuple(
      std::move(grad_query), std::move(grad_key), std::move(grad_value));
#endif
}
} // namespace sycltla
