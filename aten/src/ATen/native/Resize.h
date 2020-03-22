#pragma once

#include <ATen/ATen.h>
#include <TH/THTensor.hpp>

namespace at { namespace native {

// These functions are called by native::resize_ as well as (legacy) TH resize.
// They are not in TH/THTensor.cpp because the at namespace is easier
// to benchmark than TH; I can't get gbenchmark to call fns from THTensor.cpp

static inline void maybe_resize_storage_cpu(TensorImpl* self, int64_t new_size) {
  // It does not make sense to try to resize a storage
  // to hold 0 elements, and this can break
  // if storage_offset is positive but
  // new_size is 0, so just bail in that case
  // (same comment is in Resize.cuh)
  if (new_size > 0) {
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
    IntArrayRef size,
    c10::optional<IntArrayRef> stride) {
  if (self->sizes() == size && (!stride || self->strides() == stride)) {
    return self;
  }

  int64_t storage_size = 1;
  if (stride) {
    self->set_sizes_and_strides(size, *stride);
    // NB: storage size can be different from numel.
    for (size_t dim = 0; dim < size.size(); ++dim) {
      // FIXME: Don't rely on storage_size being negative because this
      // may not be true for some edge cases.
      if (size[dim] == 0) {
        storage_size = 0;
        break;
      }
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
    IntArrayRef size,
    IntArrayRef stride,
    int64_t storage_offset,
    const Storage& new_storage) {
  int64_t storage_size = detail::computeStorageSize(size, stride);
  if (storage_size == 0) {
    // NB: (a tensor with arbitrary 0 dims)'s storage can have any numel.
    return;
  }
  int64_t new_storage_size = new_storage.numel();
  TORCH_CHECK(
      storage_offset + storage_size <= new_storage_size,
      "setStorage: sizes ", size, ", strides ", stride, ","
      " and storage offset ", storage_offset,
      " requiring a storage size of ", storage_size + storage_offset,
      " are out of bounds for storage with numel ", new_storage_size);
}

static inline void checkSetStorage(Tensor& result, Storage storage, int64_t storage_offset,
                                   IntArrayRef size, IntArrayRef stride) {
  // FIXME: stride should be optional
  if (stride.data()) {
    TORCH_CHECK(size.size() == stride.size(), "unequal size length (", size.size(),
                                              ") and stride length (", stride.size(), ")");
  }

#ifdef DEBUG
  TORCH_CHECK(size.size() <= INT_MAX, "size length (", size.size(), ") greater than INT_MAX");
#endif

  // storage: note this can't be replaced with result.set_(storage) as the semantics of that
  // function is to set the tensor size to be equal to the size of the storage.
  if (!result.storage().is_alias_of(storage)) {
    // Caffe2 might have tensors whose storages are null, but we
    // don't allow it in PyTorch.
    TORCH_INTERNAL_ASSERT(storage);
    TORCH_INTERNAL_ASSERT(result.storage());

    // Caffe2 also has uninitialized dtype states, which we disallow here
    TORCH_INTERNAL_ASSERT(result.storage().dtype() == storage.dtype());

    // We used to allow this, but this breaks device caching.
    // Let's put an actual error message for this one.
    TORCH_CHECK(result.storage().device() == storage.device(),
                "Attempted to set the storage of a tensor on device \"", result.storage().device(),
                "\" to a storage on different device \"", storage.device(),
                "\".  This is no longer allowed; the devices must match.");
    result.unsafeGetTensorImpl()->set_storage(storage);
  }

  // storageOffset
  if (storage_offset < 0) {
    TORCH_CHECK("Tensor: invalid storage offset ", storage_offset);
  }
}

/**
 * Set self's sizes, strides, and storage_offset.
 * (size, stride, storage_offset) must be in bounds for self's storage.
 */
inline void setStrided(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    int64_t storage_offset) {
  auto* self_ = self.unsafeGetTensorImpl();
  checkInBoundsForStorage(size, stride, storage_offset, self_->storage());

  /* storage offset */
  TORCH_CHECK(storage_offset >= 0, "Tensor: invalid storage offset ", storage_offset);
  self_->set_storage_offset(storage_offset);

  /* size and stride */
  AT_ASSERT(size.size() == stride.size());
  if (self_->sizes() == size && self_->strides() == stride) {
    return;
  }
  self_->set_sizes_and_strides(size, stride);
}

}}
