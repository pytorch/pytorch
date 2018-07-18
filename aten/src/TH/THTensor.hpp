#pragma once

// STOP!!! Thinking of including this header directly?  Please
// read Note [TH abstraction violation]

#include "THTensor.h"
#include "THStorageFunctions.hpp"

#include <atomic>
#include <ATen/ATen.h>

struct THTensor
{
    THTensor(THStorage* storage)
      : refcount(1)
      , storage(storage)
      , storageOffset(0)
      , sizes_{0}
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

    std::vector<int64_t> sizes_;
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
        if (sizes_[i] == 0) {
          return true;
        }
      }
      return false;
    }

    int64_t size(int64_t dim) const {
      return sizes_[dim];
    }

    inline at::IntList sizes() {
      return sizes_;
    }

    inline at::IntList strides() {
      return stride;
    }
};

#include "generic/THTensorFastGetSet.hpp"
#include "THGenerateAllTypes.h"

inline int64_t* THTensor_getSizePtr(THTensor* tensor) {
  AT_ASSERT(static_cast<int64_t>(tensor->sizes_.size()) == tensor->dim_);
  return tensor->sizes_.data();
}

inline int64_t* THTensor_getStridePtr(THTensor* tensor) {
  AT_ASSERT(static_cast<int64_t>(tensor->stride.size()) == tensor->dim_);
  return tensor->stride.data();
}

inline void THTensor_resizeDim(THTensor* tensor, int64_t ndim) {
  tensor->dim_ = ndim;
  // NB: This is *truly* a resize; calling code (e.g., squeeze)
  // assumes that old values are preserved
  tensor->sizes_.resize(ndim);
  tensor->stride.resize(ndim);
}

inline void THTensor_setSizesAndStrides(THTensor* tensor, std::vector<int64_t>&& new_size, std::vector<int64_t>&& new_stride) {
  AT_ASSERT(new_size.size() == new_stride.size());
  tensor->dim_ = new_size.size();
  tensor->sizes_ = std::move(new_size);
  tensor->stride = std::move(new_stride);
}

inline void THTensor_setSizeAtDim(THTensor* tensor, int dim, int64_t new_size) {
  tensor->sizes_[dim] = new_size;
}

inline void THTensor_setStrideAtDim(THTensor* tensor, int dim, int64_t new_stride) {
  tensor->stride[dim] = new_stride;
}

TH_API void THTensor_free(THTensor *self);
at::optional<std::vector<int64_t>> THTensor_compute_stride(at::IntList oldshape, at::IntList oldstride,
                                                           at::IntList newshape);
