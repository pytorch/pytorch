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

  int64_t storage_size = 1;
  if (stride) {
    self->set_sizes_and_strides(size, *stride);
    // NB: storage size can be different from numel.
    for (size_t dim = 0; dim < size.size(); ++dim) {
      // FIXME: Don't rely on storage_size being negative...
      // This behavior was carried over from TH
      storage_size += (size[dim] - 1) * stride.value()[dim];
    }
  } else {
    self->set_sizes_contiguous(size);
    storage_size = self->numel();
  }
  maybe_resize_storage_cpu(self, storage_size);

  return self;
}

static inline void checkInBoundsForStorage(
    IntList size,
    IntList stride,
    ptrdiff_t storageOffset,
    StorageImpl* new_storage) {
  ptrdiff_t storage_size = 1;
  for (size_t dim = 0; dim < size.size(); ++dim) {
    // NB: (a tensor with arbitrary 0 dims)'s storage can have any numel.
    if (size[dim] == 0) {
      return;
    }
    storage_size += (size[dim] - 1) * stride[dim];
  }
  AT_CHECK(
      storageOffset + storage_size <= new_storage->numel(),
      "setStorage: sizes ", size, ", strides ", stride, ","
      " and storage offset ", storageOffset,
      " requiring a storage size of ", storage_size + storageOffset,
      " are out of bounds for storage with numel ", new_storage->numel());
}

/**
 * Set self's storage to be new_storage with sizes, strides, and storageOffset.
 * (size, stride, storageOffset) must be in bounds for the new storage.
 */
inline void setStorage(
    TensorImpl* self,
    StorageImpl* new_storage,
    ptrdiff_t storageOffset,
    IntList size,
    IntList stride) {
  AT_ASSERT(new_storage);
  checkInBoundsForStorage(size, stride, storageOffset, new_storage);

  /* storage */
  auto* old_storage = self->storage_.unsafeGetStorageImpl();
  AT_CHECK(old_storage, "Tensor: invalid null storage");
  if (old_storage != new_storage) {
    c10::raw::intrusive_ptr::incref(new_storage);
    THTensor_stealAndSetStoragePtr(self, new_storage);
  }

  /* storage offset */
  AT_CHECK(storageOffset >= 0, "Tensor: invalid storage offset ", storageOffset);
  self->set_storage_offset(storageOffset);

  /* size and stride */
  AT_ASSERT(size.size() == stride.size());
  if (self->sizes() == size && self->strides() == stride) {
    return;
  }
  self->set_sizes_and_strides(size, stride);
}

}}
