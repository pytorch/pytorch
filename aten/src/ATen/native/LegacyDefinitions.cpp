#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/LegacyTHFunctionsCPU.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/core/EnableNamedTensor.h>

namespace at { namespace native {

// Methods

Tensor & masked_fill__cpu(Tensor& self, const Tensor & mask, Scalar value) {
#ifdef BUILD_NAMEDTENSOR
  auto outnames = namedinference::broadcast_to_outnames(self, mask, "masked_fill_");
#endif
  // As we dispatch on self and TH is type-checked, we need different definitions.
  // This can be fixed by moving to ATen.
  if (mask.dtype() == at::ScalarType::Byte) {
    AT_WARN("masked_fill_ received a mask with dtype torch.uint8, this behavior is now deprecated," \
            "please use a mask with dtype torch.bool instead.");
    legacy::cpu::_th_masked_fill_(self, mask, value);
  } else {
    legacy::cpu::_th_masked_fill_bool_(self, mask, value);
  }
#ifdef BUILD_NAMEDTENSOR
  namedinference::propagate_names(self, std::move(outnames), /*validate_names=*/false);
#endif
  return self;
}

Tensor & masked_fill__cpu(Tensor& self, const Tensor & mask, const Tensor & value) {
#ifdef BUILD_NAMEDTENSOR
  auto outnames = namedinference::broadcast_to_outnames(self, mask, "masked_fill_");
#endif

  TORCH_CHECK(value.dim() == 0, "masked_fill_ only supports a 0-dimensional value tensor, but got tensor "
      "with ", value.dim(), " dimension(s).");
  // As we dispatch on self and TH is type-checked, we need different definitions.
  // This can be fixed by moving to ATen.
  if (mask.dtype() == at::ScalarType::Byte) {
    AT_WARN("masked_fill_ received a mask with dtype torch.uint8, this behavior is now deprecated," \
            "please use a mask with dtype torch.bool instead.");
    legacy::cpu::_th_masked_fill_(self, mask, value.item());
  } else {
    legacy::cpu::_th_masked_fill_bool_(self, mask, value.item());
  }
#ifdef BUILD_NAMEDTENSOR
  namedinference::propagate_names(self, std::move(outnames), /*validate_names=*/false);
#endif
  return self;
}

Tensor & masked_scatter__cpu(Tensor& self, const Tensor & mask, const Tensor & source) {
  // As we dispatch on self and TH is type-checked, we need different definitions.
  // This can be fixed by moving to ATen.
  if (mask.dtype() == at::ScalarType::Byte) {
    AT_WARN("masked_scatter_ received a mask with dtype torch.uint8, this behavior is now deprecated," \
            "please use a mask with dtype torch.bool instead.");
    return legacy::cpu::_th_masked_scatter_(self, mask, source);
  } else {
    return legacy::cpu::_th_masked_scatter_bool_(self, mask, source);
  }
}

Tensor masked_select_cpu(const Tensor & self, const Tensor & mask) {
#ifdef BUILD_NAMEDTENSOR
  namedinference::compute_broadcast_outnames(self, mask);
#endif
  if (mask.dtype() == at::ScalarType::Byte) {
    AT_WARN("masked_select received a mask with dtype torch.uint8, this behavior is now deprecated," \
            "please use a mask with dtype torch.bool instead.");
    return legacy::cpu::_th_masked_select(self, mask);
  } else {
    return legacy::cpu::_th_masked_select_bool(self, mask);
  }
}

Tensor & masked_select_out_cpu(Tensor & result, const Tensor & self, const Tensor & mask) {
#ifdef BUILD_NAMEDTENSOR
  namedinference::compute_broadcast_outnames(self, mask);
#endif
  if (mask.dtype() == at::ScalarType::Bool) {
    return legacy::cpu::_th_masked_select_bool_out(result, self, mask);
  } else {
    return legacy::cpu::_th_masked_select_out(result, self, mask);
  }
}

Tensor argsort(const Tensor & self, int64_t dim, bool descending) {
  return std::get<1>(at::sort(self, dim, descending));
}

Tensor & gather_out_cpu(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  return legacy::cpu::_th_gather_out(result, self, dim, index);
}

Tensor gather_cpu(const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  return legacy::cpu::_th_gather(self, dim, index);
}

}} // namespace at::native
