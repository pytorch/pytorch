#pragma once

#include <ATen/ATen.h>
#include <ATen/native/ResizeCommon.h>
#include <ATen/TensorUtils.h>
#include <TH/THTensor.hpp>


namespace at { namespace native {

// TODO: make all operations that resize given outputs use this function
//   for consistency and maintainability
// Resizes outputs
// Functions accepting output tensors, like with the "out" kwarg, should
//   call this function to handle resizing their output tensor.
// Issues a warning if the output tensor has one or more elements and
//   needs resizing
// NOTE: In the future the warning will become an error
// Returns a bool saying whether or not the resize actually happened or not
TORCH_API bool resize_output(const Tensor& output, IntArrayRef shape);

// These functions are called by native::resize_ as well as (legacy) TH resize.
// They are not in TH/THTensor.cpp because the at namespace is easier
// to benchmark than TH; I can't get gbenchmark to call fns from THTensor.cpp

static inline void maybe_resize_storage_cpu(TensorImpl* self, uint64_t new_size) {
  // It does not make sense to try to resize a storage
  // to hold 0 elements, and this can break
  // if storage_offset is positive but
  // new_size is 0, so just bail in that case
  // (same comment is in Resize.cuh)
  if (new_size == 0) {
    return;
  }
  if (!THTensor_getStoragePtr(self)) {
#ifndef NDEBUG
    caffe2::TypeMeta dtype = self->dtype();
#endif
    THTensor_stealAndSetStoragePtr(self, THStorage_new());
#ifndef NDEBUG
    // THTensor_stealAndSetStoragePtr guarantees this. Leave debug
    // assert in case of code changes.
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dtype == self->dtype());
#endif
  }
  uint64_t new_size_bytes =
      (new_size + self->storage_offset()) * self->dtype().itemsize();
  if (new_size_bytes > self->storage().nbytes()) {
    THStorage_resizeBytes(THTensor_getStoragePtr(self), new_size_bytes);
  }
}

inline TensorImpl* resize_impl_cpu_(
    TensorImpl* self,
    IntArrayRef size,
    c10::optional<IntArrayRef> stride,
    bool resize_storage = true) {
  if (self->sizes() == size && (!stride || self->strides() == stride)) {
    return self;
  }

  int64_t storage_size = 1;
  if (stride) {
    self->set_sizes_and_strides(size, *stride);
    // NB: storage size can be different from numel.
    storage_size = storage_size_for(size, *stride);
  } else {
    self->set_sizes_contiguous(size);
    storage_size = self->numel();
  }
  if (resize_storage) {
    maybe_resize_storage_cpu(self, storage_size);
  }

  return self;
}

static inline void checkInBoundsForStorage(
    IntArrayRef size,
    IntArrayRef stride,
    int64_t storage_offset,
    const caffe2::TypeMeta data_type,
    const Storage& new_storage) {
  int64_t storage_size_bytes =
      at::detail::computeStorageNbytes(size, stride, data_type.itemsize());
  int64_t storage_offset_bytes = storage_offset * data_type.itemsize();
  if (storage_size_bytes == 0) {
    // NB: (a tensor with arbitrary 0 dims)'s storage can have any numel.
    return;
  }
  int64_t new_storage_size_bytes = new_storage.nbytes();
  TORCH_CHECK(
      storage_size_bytes + storage_offset_bytes <= new_storage_size_bytes,
      "setStorage: sizes ",
      size,
      ", strides ",
      stride,
      ","
      " storage offset ",
      storage_offset,
      ", and itemsize ",
      data_type.itemsize(),
      " requiring a storage size of ",
      storage_size_bytes + storage_offset_bytes,
      " are out of bounds for storage of size ",
      new_storage_size_bytes);
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

    // We used to allow this, but this breaks device caching.
    // Let's put an actual error message for this one.
    TORCH_CHECK(result.storage().device() == storage.device(),
                "Attempted to set the storage of a tensor on device \"", result.storage().device(),
                "\" to a storage on different device \"", storage.device(),
                "\".  This is no longer allowed; the devices must match.");
    result.unsafeGetTensorImpl()->set_storage_keep_dtype(storage);
  }

  // storageOffset
  TORCH_CHECK(storage_offset >= 0, "Tensor: invalid storage offset ", storage_offset);
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
  TORCH_CHECK(size.size() == stride.size(), "mismatch in length of strides and shape");
  auto* self_ = self.unsafeGetTensorImpl();
  checkInBoundsForStorage(
      size, stride, storage_offset, self_->dtype(), self_->storage());

  /* storage offset */
  TORCH_CHECK(storage_offset >= 0, "Tensor: invalid storage offset ", storage_offset);
  self_->set_storage_offset(storage_offset);

  /* size and stride */
  if (self_->sizes() == size && self_->strides() == stride) {
    return;
  }
  for (auto val : stride) {
    TORCH_CHECK(val >= 0,
                "as_strided: Negative strides are not supported at the moment, "
                "got strides: ", stride);
  }
  self_->set_sizes_and_strides(size, stride);
}

}}
