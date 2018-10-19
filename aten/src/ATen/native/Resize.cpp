#include "ATen/ATen.h"
#include "TH/THTensor.hpp"

namespace at { namespace native {

Tensor& resize__cpu(Tensor& self, IntList size) {
  if (self.sizes() == size) {
    return self;
  }

  auto* self_ = self.unsafeGetTensorImpl();
  self_->set_sizes_contiguous(size);
  int totalSize = self_->numel();

  if (totalSize + self_->storage_offset() > 0) {
    if (!THTensor_getStoragePtr(self_)) {
      THTensor_stealAndSetStoragePtr(self_, THStorage_new(self_->dtype()));
    }
    if (totalSize + self_->storage_offset() > self_->storage().numel()) {
      THStorage_resize(
          THTensor_getStoragePtr(self_),
          totalSize + self_->storage_offset());
    }
  }
  self_->maybe_zero_dim(size.size() == 0);

  return self;
}

}}
