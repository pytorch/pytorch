#include <c10/util/Exception.h>
#include <ATen/ATen.h>
#include <ATen/Functions.h>

namespace at::native {

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

Tensor relu_quantized_cuda(const Tensor& self) {
  auto zero_point = self.q_zero_point();
  auto int_repr = self.int_repr();
  auto mask = (int_repr > zero_point);
  const auto relu_int_repr = at::where(mask, int_repr, zero_point);
  return at::_make_per_tensor_quantized_tensor(relu_int_repr, self.q_scale(), zero_point);
}

}  // namespace at::native
