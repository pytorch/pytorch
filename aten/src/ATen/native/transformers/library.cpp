#include <ATen/native/transformers/library.h>
#include <ATen/native/nested/NestedTensorTransformerFunctions.h>
#include <ATen/native/transformers/transformer.h>

#include <torch/library.h>

namespace at {
namespace native {
TORCH_LIBRARY(nativetransformers, m) {
  m.def(
      "_native_multi_head_attention(Tensor query, Tensor key, Tensor value, "
      "int embed_dim, int num_head, Tensor qkv_weight, Tensor qkv_bias, "
      "Tensor proj_weight, Tensor proj_bias, Tensor? mask=None, "
      "bool need_weights=True, bool average_attn_weights=True) -> (Tensor, Tensor)",
      TORCH_FN(multi_head_attention));
  m.impl(
      "_native_multi_head_attention",
      c10::DispatchKey::NestedTensorCPU,
      TORCH_FN(multi_head_attention));
  m.impl(
      "_native_multi_head_attention",
      c10::DispatchKey::NestedTensorCUDA,
      TORCH_FN(multi_head_attention));

  m.def(
      "_transform_bias_rescale_qkv_op(Tensor qkv, Tensor qkv_bias, int num_head) -> (Tensor, Tensor, Tensor)");
  m.impl(
      "_transform_bias_rescale_qkv_op",
      c10::kCPU,
      TORCH_FN(transform_bias_rescale_qkv_op_cpu));

  m.def(
      "_ffn(Tensor input, Tensor weight1, Tensor bias1, Tensor weight2, Tensor bias2, bool use_gelu=False, bool add_norm=False) -> Tensor");
  m.impl("_ffn", c10::kCPU, TORCH_FN(ffn_cpu));

  // TODO Current ffn for cpu and cuda shares the same slow kernel.
  // We need to fix here when we have separate kernels.
  m.impl("_ffn", c10::kCUDA, TORCH_FN(ffn_cpu));

  m.def(
      "_transformer_encoder_layer_forward(Tensor src, int embed_dim, "
      "int num_heads, Tensor qkv_weight, Tensor qkv_bias, Tensor proj_weight, "
      "Tensor proj_bias, bool use_gelu, bool norm_first, "
      "float eps, Tensor norm_weight_1, Tensor norm_bias_1, Tensor "
      "norm_weight_2, Tensor norm_bias_2, Tensor ffn_weight_1, Tensor "
      "ffn_bias_1, Tensor ffn_weight_2, Tensor ffn_bias_2, Tensor? mask=None) -> Tensor",
      TORCH_FN(transformer_encoder_layer_forward));
  m.impl(
      "_transformer_encoder_layer_forward",
      c10::DispatchKey::NestedTensor,
      TORCH_FN(transformer_encoder_layer_forward));

  m.def("_to_padded_tensor(Tensor nt, float padding) -> Tensor");
  m.impl(
      "_to_padded_tensor",
      c10::DispatchKey::NestedTensor,
      TORCH_FN(NestedTensor_to_padded_tensor));

  m.def(
      "_layer_norm(Tensor nt, Tensor? weight, Tensor? bias, float eps) -> Tensor");
  m.impl(
      "_layer_norm",
      c10::DispatchKey::NestedTensor,
      TORCH_FN(NestedTensor_layer_norm));
}
} // namespace native
} // namespace at
