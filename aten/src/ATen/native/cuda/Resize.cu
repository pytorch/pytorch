#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <ATen/native/cuda/Resize.cuh>

namespace at { namespace native {

Tensor& resize_cuda_(Tensor& self, IntArrayRef size) {
  auto* self_ = self.unsafeGetTensorImpl();
  resize_impl_cuda_(self_, size, /*strides=*/c10::nullopt);
  self_->maybe_zero_dim(size.size() == 0);
  return self;
}

Tensor& resize_as_cuda_(Tensor& self, const Tensor& the_template) {
  return resize_cuda_(self, the_template.sizes());
}

}}
