#include <c10/util/Exception.h>
#include <ATen/ATen.h>

namespace at {
namespace native {

// this kernel is currently implemented with dequantize -> fp32 gelu -> quantize, which is not equivalent to int8 gelu
// It might be possible to write a variant of the int8 gelu that's equivalent to dequantize -> fp32 cuda gelu kernel -> quantize,
// which can be a topic for future work.
Tensor gelu_quantized_cuda(const Tensor& qx, c10::string_view approximate) {
  (void)approximate; // suppress unused variable lint warning
  if (qx.numel() == 0) {
    return Tensor{};
  }
  auto x_fp32 = at::dequantize(qx);
  auto result_fp32 = at::gelu(x_fp32);
  return at::quantize_per_tensor(result_fp32, qx.q_scale(), qx.q_zero_point(), qx.scalar_type());
}

}  // namespace at::native
}  // namespace at
