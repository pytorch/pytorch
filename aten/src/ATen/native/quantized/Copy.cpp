#include <ATen/native/Copy.h>

#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/quantized/Quantizer.h>

namespace at {
namespace native {
Tensor& _s_copy__quantized(Tensor& self, const Tensor& src, bool /* unused */) {
  AT_CHECK(
      self.scalar_type() == at::kQUInt8,
      "Quantized copy only works with kQUInt8 as target Tensor");
  AT_CHECK(
      src.scalar_type() == at::kFloat,
      "Quantized copy only works with kFloat as source Tensor");
  quint8* self_data = self.data<quint8>();
  float* src_data = src.data<float>();
  for (int i = 0; i < self.numel(); ++i) {
    // TODO: DISPATCH
    self_data[i] = quantize_uint<quint8>(
        self.q_scale().to<float>(),
        self.q_zero_point().to<int32_t>(),
        src_data[i]);
  }
  return self;
}
} // namespace native
} // namespace at
