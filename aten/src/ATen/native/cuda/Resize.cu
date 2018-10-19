#include "ATen/ATen.h"
#include "ATen/cuda/CUDAContext.h"

#include "THC/THCTensor.hpp"

namespace at { namespace native {

Tensor& resize__cuda(Tensor& self, IntList size) {
  if (self.sizes() == size) {
    return self;
  }

  const DeviceGuard device_guard(self);
  auto* self_ = self.unsafeGetTensorImpl();
  self_->set_sizes_contiguous(size);
  int totalSize = self_->numel();

  if (totalSize + self_->storage_offset() > 0) {
    if (!THTensor_getStoragePtr(self_)) {
      AT_ERROR("Tensor: invalid null storage");
    }
    if (totalSize + self_->storage_offset() > self_->storage().numel()) {
      THCStorage_resize(
          globalContext().getTHCState(),
          THTensor_getStoragePtr(self_),
          totalSize + self_->storage_offset());
    }
  }
  self_->maybe_zero_dim(size.size() == 0);

  return self;
}

}}
