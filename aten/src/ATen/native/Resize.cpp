#include <ATen/ATen.h>
#include <ATen/native/Resize.h>

namespace at { namespace native {

Tensor& resize_cpu_(Tensor& self, IntArrayRef size) {
  auto* self_ = self.unsafeGetTensorImpl();
  resize_impl_cpu_(self_, size, /*strides=*/c10::nullopt);
  self_->maybe_zero_dim(size.size() == 0);
  return self;
}

}}
