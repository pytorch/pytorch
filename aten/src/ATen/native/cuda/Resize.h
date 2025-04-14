#pragma once

#include <ATen/EmptyTensor.h>
#include <ATen/native/ResizeCommon.h>

#include <c10/cuda/CUDAGuard.h>

namespace at::native {

TORCH_CUDA_CPP_API void resize_bytes_cuda(StorageImpl* storage, size_t size_bytes);

static inline void maybe_resize_storage_cuda(TensorImpl* self, size_t new_size_bytes) {
  // It does not make sense to try to resize a storage
  // to hold 0 elements, and this can break
  // if storage_offset is positive but
  // new_size is 0, so just bail in that case
  // (same comment is in Resize.h)
  if (self->numel() == 0) {
    return;
  }

  const Storage &storage = self->unsafe_storage();
  TORCH_CHECK(storage, "Tensor: invalid null storage");
  if (new_size_bytes > storage.nbytes()) {
    resize_bytes_cuda(storage.unsafeGetStorageImpl(), new_size_bytes);
  }
}

inline TensorImpl* resize_impl_cuda_(
    TensorImpl* self,
    IntArrayRef size,
    at::OptionalIntArrayRef stride) {
  if (self->sizes() == size && (!stride || self->strides() == stride)) {
    return self;
  }
  const auto itemsize = self->dtype().itemsize();
  const auto storage_offset = self->storage_offset();
  size_t storage_size = 1;
  if (stride) {
    self->set_sizes_and_strides(size, *stride);
    storage_size = at::detail::computeStorageNbytes(
        size, *stride, itemsize, storage_offset);
  } else {
    self->set_sizes_contiguous(size);
    storage_size = at::detail::computeStorageNbytesContiguous(
        size, itemsize, storage_offset);
  }
  maybe_resize_storage_cuda(self, storage_size);

  return self;
}

}
