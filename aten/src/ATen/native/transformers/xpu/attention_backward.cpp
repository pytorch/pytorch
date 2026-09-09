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

std::tuple<at::Tensor, at::Tensor, at::Tensor>
_flash_attention_backward_xpu(
    const at::Tensor& grad_out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const at::Tensor& logsumexp,
    const at::Tensor& cum_seq_q,
    const at::Tensor& cum_seq_k,
    int64_t max_q,
    int64_t max_k,
    double dropout_p,
    bool is_causal,
    const at::Tensor& rng_state,
    const at::Tensor& unused,
    std::optional<double> scale,
    std::optional<int64_t> window_size_left,
    std::optional<int64_t> window_size_right) {
  if (!grad_out.defined()) {
    return std::make_tuple(Tensor{}, Tensor{}, Tensor{});
  }

  TORCH_CHECK(
      !window_size_left.has_value() && !window_size_right.has_value(),
      "_flash_attention_backward: window_size_left and window_size_right are not supported on XPU");
  TORCH_CHECK(
      dropout_p == 0.0,
      "_flash_attention_backward: dropout is not yet properly supported on XPU (RNG state handling not implemented)");

  // Validate dtype early for better error messages
  auto dtype = query.scalar_type();
  TORCH_CHECK(
      dtype == at::kHalf || dtype == at::kBFloat16,
      "_flash_attention_backward: only fp16 and bf16 data types are supported on XPU, got ",
      dtype);

  // Varlen (nested tensor) backward is not yet supported on XPU.
  // Detect varlen by checking if cum_seq_q or cum_seq_k are defined.
  bool is_varlen = cum_seq_q.defined() || cum_seq_k.defined();
  TORCH_CHECK(
      !is_varlen,
      "_flash_attention_backward: varlen (nested tensor) backward is not yet "
      "supported on XPU. cum_seq_q and cum_seq_k must be undefined.");

  const float scale_val = scale.has_value()
      ? static_cast<float>(scale.value())
      : static_cast<float>(1.0 / std::sqrt(query.size(3)));

  // Delegate to existing dense backward implementation.
  // The existing sycltla backward takes philox_seed/philox_offset separately.
  // Dropout is rejected above, so these tensors are placeholders for the
  // required arguments in the no-dropout path. Initialize them to zero rather
  // than using uninitialized storage so correctness does not depend on the
  // callee never touching them.
  auto philox_seed = at::zeros({}, at::dtype(at::kLong).device(query.device()));
  auto philox_offset = at::zeros({}, at::dtype(at::kLong).device(query.device()));

  auto [grad_q, grad_k, grad_v] = sycltla::flash_attention_backward(
      grad_out,
      query,
      key,
      value,
      out,
      logsumexp,
      cum_seq_q,
      cum_seq_k,
      max_q,
      max_k,
      dropout_p,
      is_causal,
      philox_seed,
      philox_offset,
      scale_val);

  return std::make_tuple(
      std::move(grad_q), std::move(grad_k), std::move(grad_v));
}

} // namespace native
} // namespace at
