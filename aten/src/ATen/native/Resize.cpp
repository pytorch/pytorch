#include "ATen/ATen.h"
#include "TH/THTensor.hpp"

namespace at { namespace native {

Tensor& resize__cpu(Tensor& self, IntList size) {
  if (self.sizes() == size) {
    return self;
  }

  auto* self_ = self.unsafeGetTensorImpl();

  int totalSize = 1;
  int nDimension = size.size();
  if(nDimension != self_->dim()) {
    self_->resize_dim(nDimension);
  }

  for (int d = nDimension - 1; d >= 0; d--) {
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
      THTensor_stealAndSetStoragePtr(self_, THStorage_new(self_->dtype()));
    }
    if (totalSize + self_->storage_offset() > THTensor_getStoragePtr(self_)->numel()) {
      THStorage_resize(THTensor_getStoragePtr(self_), totalSize+self_->storage_offset());
    }
  }
  self_->maybe_zero_dim(size.size() == 0);

  return self;
}

}}
