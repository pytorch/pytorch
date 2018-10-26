#pragma once

#include "ATen/ATen.h"
#include "TH/THTensor.hpp"

namespace at { namespace native {

// These functions are called by native::resize_ as well as (legacy) TH resize.
// They are not in TH/THTensor.cpp because the at namespace is easier
// to benchmark than TH; I can't get gbenchmark to call fns from THTensor.cpp

static inline void maybe_resize_storage_cpu(TensorImpl* self, int64_t new_size) {
  if (new_size + self->storage_offset() > 0) {
    if (!THTensor_getStoragePtr(self)) {
      THTensor_stealAndSetStoragePtr(self, THStorage_new(self->dtype()));
    }
    if (new_size + self->storage_offset() > self->storage().numel()) {
      THStorage_resize(
          THTensor_getStoragePtr(self),
          new_size + self->storage_offset());
    }
  }
}

inline TensorImpl* resize_impl_cpu_(
    TensorImpl* self,
    IntList size,
    c10::optional<IntList> stride) {
  if (self->sizes() == size && (!stride || self->strides() == stride)) {
    return self;
  }

  size_t storage_size = 1;
  if (stride) {
    self->set_sizes_and_strides(size, *stride);
    // NB: storage size can be different from numel.
    for (size_t dim = 0; dim < size.size(); ++dim) {
      storage_size += (size[dim] - 1) * stride.value()[dim];
    }
  } else {
    self->set_sizes_contiguous(size);
    storage_size = self->numel();
  }
  maybe_resize_storage_cpu(self, storage_size);

  return self;
}

}}
