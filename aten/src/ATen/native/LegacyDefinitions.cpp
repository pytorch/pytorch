#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/LegacyTHFunctionsCPU.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>

namespace at { namespace native {

// Methods

Tensor & masked_scatter__cpu(Tensor& self, const Tensor & mask, const Tensor & source) {
  at::assert_no_internal_overlap(self);
  Tensor b_mask;
  std::tie(b_mask) = expand_inplace(self, mask, "masked_scatter_");
  // As we dispatch on self and TH is type-checked, we need different definitions.
  // This can be fixed by moving to ATen.
  if (b_mask.dtype() == at::ScalarType::Byte) {
    TORCH_WARN("masked_scatter_ received a mask with dtype torch.uint8, this behavior is now deprecated," \
            "please use a mask with dtype torch.bool instead.");
    return legacy::cpu::_th_masked_scatter_(self, b_mask, source);
  } else {
    return legacy::cpu::_th_masked_scatter_bool_(self, b_mask, source);
  }
}

Tensor argsort(const Tensor & self, int64_t dim, bool descending) {
  return std::get<1>(at::sort(self, dim, descending));
}

}} // namespace at::native
