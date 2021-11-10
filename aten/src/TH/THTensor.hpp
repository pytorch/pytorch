#pragma once

// STOP!!! Thinking of including this header directly?  Please
// read Note [TH abstraction violation]

#include <TH/THTensor.h>
#include <TH/THStorageFunctions.hpp>

#include <atomic>
#include <ATen/ATen.h>

// Returns a Tensor given a TensorImpl. The TensorImpl remains valid after the
// the Tensor is destroyed.
inline at::Tensor THTensor_wrap(THTensor* tensor) {
  c10::raw::intrusive_ptr::incref(tensor);
  return at::Tensor(c10::intrusive_ptr<at::TensorImpl>::reclaim(tensor));
}

TH_API void THTensor_free(THTensor *self);

TH_CPP_API void THTensor_setStorage(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_, at::IntArrayRef size_, at::IntArrayRef stride_);
