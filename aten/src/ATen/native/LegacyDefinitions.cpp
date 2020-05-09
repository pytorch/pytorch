#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/LegacyTHFunctionsCPU.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/ExpandUtils.h>

namespace at { namespace native {

// Methods

Tensor & masked_scatter__cpu(Tensor& self, const Tensor & mask, const Tensor & source) {
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

Tensor masked_select_cpu(const Tensor & self, const Tensor & mask) {
  namedinference::compute_broadcast_outnames(self, mask);

  Tensor b_self, b_mask;
  std::tie(b_self, b_mask) = expand_outplace(self, mask, "masked_select");
  if (b_mask.dtype() == at::ScalarType::Byte) {
    TORCH_WARN("masked_select received a mask with dtype torch.uint8, this behavior is now deprecated," \
            "please use a mask with dtype torch.bool instead.");
    return legacy::cpu::_th_masked_select(b_self, b_mask);
  } else {
    return legacy::cpu::_th_masked_select_bool(b_self, b_mask);
  }
}

Tensor & masked_select_out_cpu(Tensor & result, const Tensor & self, const Tensor & mask) {
  namedinference::compute_broadcast_outnames(self, mask);

  Tensor b_self, b_mask;
  std::tie(b_self, b_mask) = expand_outplace(self, mask, "masked_select_out");
  if (b_mask.dtype() == at::ScalarType::Bool) {
    return legacy::cpu::_th_masked_select_bool_out(result, b_self, b_mask);
  } else {
    return legacy::cpu::_th_masked_select_out(result, b_self, b_mask);
  }
}

Tensor argsort(const Tensor & self, int64_t dim, bool descending) {
  return std::get<1>(at::sort(self, dim, descending));
}

}} // namespace at::native
