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

  int totalSize = 1;
  int nDimension = size.size();
  if(nDimension != self_->dim()) {
    self_->resize_dim(nDimension);
  }

  for(int d = nDimension - 1; d >= 0; d--) {
    self_->set_size(d, size[d]);
    if(d == nDimension - 1) {
      self_->set_stride(d, 1);
    } else {
      // Keep stride monotonically increasing to match NumPy.
      self_->set_stride(d, std::max<int64_t>(self_->size(d + 1), 1) * self_->stride(d + 1));
    }
    totalSize += (self_->size(d) - 1) * self_->stride(d);
  }

  if (totalSize + self_->storage_offset() > 0) {
    if (!THTensor_getStoragePtr(self_)) {
      AT_ERROR("Tensor: invalid null stroage");
    }
    if (totalSize + self_->storage_offset() > THTensor_getStoragePtr(self_)->numel()) {
      THCStorage_resize(
          globalContext().getTHCState(),
          THTensor_getStoragePtr(self_),
          totalSize+self_->storage_offset());
    }
  }
  self_->maybe_zero_dim(size.size() == 0);

  return self;
}

}}
