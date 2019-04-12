#include <ATen/native/Copy.h>

#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>

namespace at { namespace native {
Tensor _s_copy_from_quantized(
    const Tensor& self,
    const Tensor& dst,
    bool non_blocking) {
  AT_CHECK(self.scalar_type() == at::kQInt8, "Quantized copy only works with kQInt8 as source Tensor");
  AT_CHECK(dst.scalar_type() == at::kByte, "Quantized copy only works with kByte as target Tensor");
  at::CPU_tensor_apply2<uint8_t, qint8>(
      dst, self, [](uint8_t& dst_val, const qint8& self_val) {
        dst_val = static_cast<uint8_t>(
            static_cast<at::native::inter_copy_type_t<uint8_t>>(self_val.val_));
      });
  return dst;
}
}} // at::native
