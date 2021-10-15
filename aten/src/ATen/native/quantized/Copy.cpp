#include <ATen/native/quantized/Copy.h>

#include <ATen/ATen.h>
#include <ATen/native/quantized/affine_quantizer.h>

namespace at {
namespace native {

// Copying from float to QInt, used for assigning float value to QTensor
Tensor& quantized_copy_from_float_cpu_(Tensor& self, const Tensor& src) {
  TORCH_CHECK(
      src.scalar_type() == at::kFloat,
      "Quantized copy only works with kFloat as source Tensor");
  TORCH_CHECK(
      self.is_contiguous() && src.is_contiguous(),
      "Quantized copy only works with contiguous Tensors");
  TORCH_CHECK(
      self.sizes().equals(src.sizes()),
      "Quantized copy only works with Tensors with the same shape");
  TORCH_CHECK(
      self.device().type() == kCPU,
      "Quantized copy only works with QuantizedCPU Tensors");
  AT_DISPATCH_QINT_TYPES(self.scalar_type(), "Copy", [&]() {
    float* src_data = src.data_ptr<float>();
    scalar_t* self_data = self.data_ptr<scalar_t>();
    for (int i = 0; i < self.numel(); ++i) {
      self_data[i] = quantize_val<scalar_t>(
          self.q_scale(), self.q_zero_point(), src_data[i]);
    }
  });
  return self;
}
} // namespace native
} // namespace at
