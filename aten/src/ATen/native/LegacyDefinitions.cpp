#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

namespace at { namespace native {

int64_t storage_offset(const Tensor& self) {
  return self._th_storage_offset();
}

int64_t ndimension(const Tensor& self) {
  return self._th_ndimension();
}

Tensor & resize_(Tensor& self, IntList size) {
  return self._th_resize_(size);
}

Tensor & set_(Tensor& self, Storage source) {
  return self._th_set_(source);
}

Tensor & set_(Tensor& self, Storage source, int64_t storage_offset, IntList size, IntList stride) {
  return self._th_set_(source, storage_offset, size, stride);
}

Tensor & set_(Tensor& self, const Tensor & source) {
  return self._th_set_(source);
}

Tensor & set_(Tensor& self) {
  return self._th_set_();
}

bool is_contiguous(const Tensor& self) {
  return self._th_is_contiguous();
}

bool is_set_to(const Tensor& self, const Tensor & tensor) {
  return self._th_is_set_to(tensor);
}

Tensor & masked_fill_(Tensor& self, const Tensor & mask, Scalar value) {
  return self._th_masked_fill_(mask, value);
}

Tensor & masked_fill_(Tensor& self, const Tensor & mask, const Tensor & value) {
  return self._th_masked_fill_(mask, value);
}

Tensor & masked_scatter_(Tensor& self, const Tensor & mask, const Tensor & source) {
  return self._th_masked_scatter_(mask, source);
}

Tensor view(const Tensor& self, IntList size) {
  return self._th_view(size);
}

}} // namespace at::native
