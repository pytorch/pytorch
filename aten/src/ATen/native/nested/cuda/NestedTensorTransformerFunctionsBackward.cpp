#include <ATen/ATen.h>
#include <ATen/native/nested/cuda/NestedTensorTransformerUtils.h>
#include <ATen/native/nested/NestedTensorUtils.h>

namespace at {
namespace native {

std::tuple<at::Tensor, at::Tensor, at::Tensor> _scaled_dot_product_flash_attention_backward_nested(
    const at::Tensor& grad_out_,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const at::Tensor& logsumexp,
    const Tensor& cumulative_sequence_length_q,
    const Tensor& cumulative_sequence_length_k,
    const int64_t max_seqlen_batch_q,
    const int64_t max_seqlen_batch_k,
    double dropout_p,
    bool is_causal,
    const int64_t philox_seed,
    const int64_t philox_offset,
    c10::optional<double> scale){
  if (!grad_out_.defined()) {
    return std::make_tuple(Tensor{}, Tensor{}, Tensor{});
  }
  Tensor grad_out_buffer_reshaped, query_buffer_reshaped, key_buffer_reshaped,
      value_buffer_reshaped, output_buffer_reshaped;
  std::tie(
      grad_out_buffer_reshaped,
      query_buffer_reshaped,
      key_buffer_reshaped,
      value_buffer_reshaped,
      output_buffer_reshaped) =
      preprocessing::sdpa_nested_preprocessing_backward(
          grad_out_,
          query,
          key,
          value,
          out,
          cumulative_sequence_length_q,
          cumulative_sequence_length_k,
          max_seqlen_batch_q,
          max_seqlen_batch_k);

  Tensor grad_q, grad_k, grad_v;
  std::tie(grad_q, grad_k, grad_v) = at::_flash_attention_backward(
    grad_out_buffer_reshaped,
    query_buffer_reshaped,
    key_buffer_reshaped,
    value_buffer_reshaped,
    output_buffer_reshaped,
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

  grad_q = wrap_buffer(grad_q.view(-1), query.transpose(1,2)._nested_tensor_size()).transpose(1,2);
  grad_k = wrap_buffer(grad_k.view(-1), key.transpose(1,2)._nested_tensor_size()).transpose(1,2);
  grad_v = wrap_buffer(grad_v.view(-1), value.transpose(1,2)._nested_tensor_size()).transpose(1,2);

  return std::make_tuple(grad_q, grad_k, grad_v);
}

} // namespace native
} // namespace at
