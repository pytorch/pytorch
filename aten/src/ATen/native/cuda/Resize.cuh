#pragma once

#include "ATen/ATen.h"
#include "THC/THCTensor.hpp"

#include "ATen/cuda/CUDAGuard.h"
#include "ATen/native/Resize.h"

namespace at { namespace native {

// These functions are called by native::resize_ as well as (legacy) THC resize.
// They are not in THC/THCTensor.cpp because the at namespace is easier
// to benchmark than THC; I can't get gbenchmark to call fns from THTensor.cpp

static inline void maybe_resize_storage_cuda(TensorImpl* self, int64_t new_size) {
  if (new_size == 0) {
    return;
  }
  if (!THTensor_getStoragePtr(self)) {
    AT_ERROR("Tensor: invalid null storage");
  }
  if (new_size > self->storage().numel()) {
    THCStorage_resize(
        globalContext().getTHCState(),
        THTensor_getStoragePtr(self),
        new_size);
  }
}

inline TensorImpl* resize_impl_cuda_(
    TensorImpl* self,
    IntList size,
    c10::optional<IntList> stride,
    bool device_guard=true) {
  if (self->sizes() == size && (!stride || self->strides() == stride)) {
    return self;
  }

  // NB: We don't need to hold the device guard when calling from TH
  cuda::OptionalCUDAGuard guard;
  if (device_guard) {
    guard.set_index(self->storage().device().index());
  }

  int64_t storage_offset = self->storage_offset();
  int64_t storage_size = 1;
  if (stride) {
    self->set_sizes_and_strides(size, *stride);
    storage_size = computeStorageSize(size, *stride, storage_offset);
  } else {
    self->set_sizes_contiguous(size);
    storage_size = self->numel() + storage_offset;
  }
  maybe_resize_storage_cuda(self, storage_size);

  return self;
}

}}
