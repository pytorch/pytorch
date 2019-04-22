#include <ATen/native/Copy.h>

#include <ATen/ATen.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/CPUApplyUtils.h>

namespace at { namespace native {
Tensor& _s_copy__quantized(
    Tensor& self,
    const Tensor& src,
    bool non_blocking) {
  AT_CHECK(self.scalar_type() == at::kQInt8, "Quantized copy only works with kQInt8 as target Tensor");
  AT_CHECK(src.scalar_type() == at::kFloat, "Quantized copy only works with kFloat as source Tensor");
  qint8* self_data = self.data<qint8>();
  float* src_data = src.data<float>();
  for (int i = 0; i < self.numel(); ++i) {
    self_data[i] = quantize_uint8(self.q_scale().to<float>(), self.q_zero_point().to<uint8_t>(), src_data[i]);
  }
  return self;
}
}} // at::native
