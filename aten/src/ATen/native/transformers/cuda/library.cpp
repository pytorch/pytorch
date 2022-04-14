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
}
} // namespace native
} // namespace at
