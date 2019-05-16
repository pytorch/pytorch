#include <ATen/native/Copy.h>

#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/quantized/Quantizer.h>

namespace at {
namespace native {
Tensor& _s_copy__quantized(Tensor& self, const Tensor& src, bool /* unused */) {
  TORCH_CHECK(
      src.scalar_type() == at::kFloat,
      "Quantized copy only works with kFloat as source Tensor");
  float* src_data = src.data<float>();
  AT_DISPATCH_QINT_TYPES(self.scalar_type(), "Copy", [&]() {
    scalar_t* self_data = self.data<scalar_t>();
    for (int i = 0; i < self.numel(); ++i) {
      self_data[i] = quantize_val<scalar_t>(
          self.q_scale().to<float>(),
          self.q_zero_point().to<int32_t>(),
          src_data[i]);
    }
  });
  return self;
}
} // namespace native
} // namespace at
