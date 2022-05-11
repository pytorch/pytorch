#include <c10/util/Exception.h>
#include <ATen/ATen.h>

namespace at {
namespace native {

// this kernel is currently implemented with dequantize -> fp32 gelu -> quantize, which is not equivalen to int8 gelu
// It might be possible to write a variant of the int8 gelu that's equivalent to dequantize -> fp32 cuda gelu kernel -> quantize,
// which can be a topic for future work. This is currently defined in the cuDNN directory because we may wish to change the
// implementation to use cudNN's gelu pointwise op
Tensor gelu_quantized_cuda(const Tensor& qx, c10::string_view approximate) {
  if (qx.numel() == 0) {
    return Tensor{};
  }
  TORCH_CHECK(qx.qscheme() == at::kPerTensorAffine, "gelu_quantized_cuda only supports per tensor quantized tensors");
  auto x_fp32 = at::dequantize(qx);
  auto result_fp32 = at::gelu(x_fp32);
  return at::quantize_per_tensor(result_fp32, qx.q_scale(), qx.q_zero_point(), qx.scalar_type());
}

}  // namespace at::native
}  // namespace at
