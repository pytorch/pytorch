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

TensorImpl* resize_impl_cuda_(
    TensorImpl* self,
    IntArrayRef size,
    c10::optional<IntArrayRef> stride,
    bool device_guard = true);

}}
