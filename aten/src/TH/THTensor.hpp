#pragma once

// STOP!!! Thinking of including this header directly?  Please
// read Note [TH abstraction violation]

#include "THTensor.h"
#include "THStorage.hpp"

#include <atomic>
#include <ATen/ATen.h>

struct THTensor
{
    THTensor(THStorage* storage)
      : refcount(1)
      , storage(storage)
      , storageOffset(0)
      , size{0}
      , stride{1}
      , dim_(1)
      {}

    ~THTensor() {
      if (storage) {
        THStorage_free(storage);
      }
    }

    std::atomic<int> refcount;

    // Note: storage->size may be greater than the recorded size
    // of a tensor
    THStorage *storage;
    ptrdiff_t storageOffset;

    std::vector<int64_t> size;
    std::vector<int64_t> stride;
    int64_t dim_;

    template <typename T>
    inline T * data() const {
      return storage->data<T>() + storageOffset;
    }

    template <typename T>
    inline T * unsafe_data() const {
      return storage->unsafe_data<T>() + storageOffset;
    }

    // [NOTE: _dim() vs dim()]
    // _dim() returns the "old" TH dimension view where no dimensions represents an empty tensor.
    // dim()  returns the ATen view of the dimensionality, i.e. 0-sized dimensions are supported.
    inline int64_t _dim() const {
      return is_empty() ? 0 : dim_;
    }

    inline int64_t dim() const {
      return dim_;
    }

    // represents that numel() == 0.
    inline bool is_empty() const {
      for (int64_t i = 0; i < dim_; ++i) {
        if (size[i] == 0) {
          return true;
        }
      }
      return false;
    }

    inline at::IntList sizes() {
      return size;
    }

    inline at::IntList strides() {
      return stride;
    }
};

#include "generic/THTensorFastGetSet.hpp"
#include "THGenerateAllTypes.h"

inline int64_t* THTensor_getSizePtr(THTensor* tensor) {
  return tensor->size.data();
}

inline int64_t* THTensor_getStridePtr(THTensor* tensor) {
  return tensor->stride.data();
}

inline void THTensor_resizeDim(THTensor* tensor, int64_t ndim) {
  tensor->size.resize(ndim);
  tensor->stride.resize(ndim);
}

inline void THTensor_setSizesAndStrides(THTensor* tensor, std::vector<int64_t>&& new_size, std::vector<int64_t>&& new_stride) {
  tensor->size = std::move(new_size);
  tensor->stride = std::move(new_stride);
}

TH_API void THTensor_free(THTensor *self);
at::optional<std::vector<int64_t>> THTensor_compute_stride(at::IntList oldshape, at::IntList oldstride,
                                                           at::IntList newshape);
