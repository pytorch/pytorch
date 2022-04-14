#include <ATen/native/transformers/library.h>
#include <ATen/native/nested/NestedTensorTransformerFunctions.h>
#include <ATen/native/transformers/transformer.h>

#include <torch/library.h>

namespace at {
namespace native {
TORCH_LIBRARY_IMPL(nativetransformers, CUDA, m) {
  m.impl(
      "_transform_bias_rescale_qkv_op",
      TORCH_FN(transform_bias_rescale_qkv_op_cuda));
}
TORCH_LIBRARY_IMPL(nativetransformers, NestedTensorCUDA, m) {
  m.impl(
      "_transform_bias_rescale_qkv_op",
      TORCH_FN(transform_bias_rescale_qkv_op_cuda));
  m.impl(
      "_transformer_encoder_layer_forward",
      TORCH_FN(transformer_encoder_layer_forward));
  m.impl(
      "_to_padded_tensor",
      TORCH_FN(NestedTensor_to_padded_tensor));
}
} // namespace native
} // namespace at
