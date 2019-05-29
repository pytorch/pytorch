#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/LegacyTHFunctionsCPU.h>

#include <ATen/native/mkldnn/TensorShape.h>

namespace at { namespace native {

// Methods

void* data_ptr(const Tensor & self) {
  return self.unsafeGetTensorImpl()->data();
}

Tensor & masked_fill__cpu(Tensor& self, const Tensor & mask, Scalar value) {
  // As we dispatch on self and TH is type-checked, we need different definitions.
  // This can be fixed by moving to ATen.
  if (mask.dtype() == at::ScalarType::Byte) {
    return legacy::cpu::_th_masked_fill_(self, mask, value);
  } else {
    return legacy::cpu::_th_masked_fill_bool_(self, mask, value);
  }
}

Tensor & masked_fill__cpu(Tensor& self, const Tensor & mask, const Tensor & value) {
  // As we dispatch on self and TH is type-checked, we need different definitions.
  // This can be fixed by moving to ATen.
  if (mask.dtype() == at::ScalarType::Byte) {
    return legacy::cpu::_th_masked_fill_(self, mask, value);
  } else {
    return legacy::cpu::_th_masked_fill_bool_(self, mask, value);
  }
}

Tensor & masked_scatter__cpu(Tensor& self, const Tensor & mask, const Tensor & source) {
  // As we dispatch on self and TH is type-checked, we need different definitions.
  // This can be fixed by moving to ATen.
  if (mask.dtype() == at::ScalarType::Byte) {
    return legacy::cpu::_th_masked_scatter_(self, mask, source);
  } else {
    return legacy::cpu::_th_masked_scatter_bool_(self, mask, source);
  }
}

Tensor masked_select_cpu(const Tensor & self, const Tensor & mask) {
  if (mask.dtype() == at::ScalarType::Byte) {
    return legacy::cpu::_th_masked_select(self, mask);
  } else {
    return legacy::cpu::_th_masked_select_bool(self, mask);
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
