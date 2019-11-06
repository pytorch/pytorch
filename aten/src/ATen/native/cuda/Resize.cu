#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <ATen/native/cuda/Resize.cuh>
#include <ATen/native/ResizeCommon.h>

namespace at { namespace native {

Tensor& resize_cuda_(Tensor& self, IntArrayRef size) {
#ifdef BUILD_NAMEDTENSOR
  if (self.has_names()) {
    return resize_named_tensor_(self, size);
  }
#endif
  auto* self_ = self.unsafeGetTensorImpl();
  resize_impl_cuda_(self_, size, /*strides=*/c10::nullopt);
  self_->maybe_zero_dim(size.size() == 0);
  return self;
}

Tensor& resize_as_cuda_(Tensor& self, const Tensor& the_template) {
  Tensor& result = resize_cuda_(self, the_template.sizes());
#ifdef BUILD_NAMEDTENSOR
  namedinference::propagate_names(result, the_template);
#endif
  return result;
}

}}
