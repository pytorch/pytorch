#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <ATen/native/transformers/xpu/flash_attn/flash_api.h>

namespace at {
namespace native {

std::tuple<Tensor, Tensor, Tensor> _flash_attention_backward_xpu(
    const Tensor& grad_out,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& out,
    const Tensor& logsumexp,
    const Tensor& cumulative_sequence_length_q,
    const Tensor& cumulative_sequence_length_k,
    int64_t max_seqlen_batch_q,
    int64_t max_seqlen_batch_k,
    double dropout_p,
    bool is_causal,
    const Tensor& philox_seed,
    const Tensor& philox_offset,
    std::optional<double> scale,
    std::optional<int64_t> window_size_left,
    std::optional<int64_t> window_size_right) {
  return sycltla::_flash_attention_backward(
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
      scale,
      window_size_left,
      window_size_right);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
_scaled_dot_product_flash_attention_backward_xpu(
    const at::Tensor& grad_out_,
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
  if (!grad_out_.defined()) {
    return std::make_tuple(Tensor{}, Tensor{}, Tensor{});
  }

  Tensor q_t = query.transpose(1, 2);
  Tensor k_t = key.transpose(1, 2);
  Tensor v_t = value.transpose(1, 2);

  Tensor grad_out_t = grad_out_.transpose(1, 2);
  Tensor out_t = out.transpose(1, 2);

  auto [grad_q, grad_k, grad_v] = at::_flash_attention_backward(
      grad_out_t,
      q_t,
      k_t,
      v_t,
      out_t,
      logsumexp,
      cumulative_sequence_length_q,
      cumulative_sequence_length_k,
      max_seqlen_batch_q,
      max_seqlen_batch_k,
      dropout_p,
      is_causal,
      philox_seed,
      philox_offset,
      scale);

  grad_q = grad_q.transpose(1, 2);
  grad_k = grad_k.transpose(1, 2);
  grad_v = grad_v.transpose(1, 2);

  return std::make_tuple(
      std::move(grad_q), std::move(grad_k), std::move(grad_v));
}

} // namespace native
} // namespace at
