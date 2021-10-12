#pragma once

#include <ATen/ATen.h>
#include <ATen/native/ResizeCommon.h>

#include <c10/cuda/CUDAGuard.h>

namespace at { namespace native {

TORCH_CUDA_CPP_API void resize_bytes_cuda(StorageImpl* storage, size_t size_bytes);

static inline void maybe_resize_storage_cuda(TensorImpl* self, uint64_t new_size) {
  // It does not make sense to try to resize a storage
  // to hold 0 elements, and this can break
  // if storage_offset is positive but
  // new_size is 0, so just bail in that case
  // (same comment is in Resize.h)
  if (new_size == 0) {
    return;
  }
  auto new_size_bytes_i = (new_size + self->storage_offset()) * self->dtype().itemsize();
  TORCH_CHECK(!overflows<size_t>(new_size_bytes_i), "Requested storage size (",
              new_size_bytes_i, ") cannot be represented as a size_t");
  const auto new_size_bytes = static_cast<size_t>(new_size_bytes_i);

  const Storage &storage = self->unsafe_storage();
  TORCH_CHECK(storage, "Tensor: invalid null storage");
  if (new_size_bytes > storage.nbytes()) {
    resize_bytes_cuda(storage.unsafeGetStorageImpl(), new_size_bytes);
  }
}

inline TensorImpl* resize_impl_cuda_(
    TensorImpl* self,
    IntArrayRef size,
    c10::optional<IntArrayRef> stride,
    bool device_guard = true) {
  if (self->sizes() == size && (!stride || self->strides() == stride)) {
    return self;
  }

  // NB: We don't need to hold the device guard when calling from TH
  cuda::OptionalCUDAGuard guard;
  if (device_guard) {
    guard.set_index(self->storage().device().index());
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
  maybe_resize_storage_cuda(self, storage_size);

  return self;
}

}}
