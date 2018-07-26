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
      : refcount_(1)
      , storage_(storage)
      , storage_offset_(0)
      , sizes_{0}
      , strides_{1}
      , is_zero_dim_(false)
      {}

    ~THTensor() {
      if (storage_) {
        THStorage_free(storage_);
      }
    }

    std::atomic<int> refcount_;

    // Note: storage->size may be greater than the recorded size
    // of a tensor
    THStorage *storage_;
    ptrdiff_t storage_offset_;

    std::vector<int64_t> sizes_;
    std::vector<int64_t> strides_;

    // TODO: get rid of this, use the sizes_/strides_ .size() instead.
    // This requires making sure TH code can handle zero dims (empty sizes, strides).
    // Short-term plan is to dispatch dim/size/stride through a function that gives these
    // in a "legacy" format, i.e. 0-dim becomes 1-dim.  Then medium term we remove the legacy calls.
    bool is_zero_dim_;

    template <typename T>
    inline T * data() const {
      return storage_->data<T>() + storage_offset_;
    }

    template <typename T>
    inline T * unsafe_data() const {
      return storage_->unsafe_data<T>() + storage_offset_;
    }

    // [NOTE: _dim() vs dim()]
    // _dim() returns the "old" TH dimension view where no dimensions represents an empty tensor.
    // dim()  returns the ATen view of the dimensionality, i.e. 0-sized dimensions are supported.
    inline int64_t _dim() const {
      return is_empty() ? 0 : dim();
    }

    inline int64_t dim() const {
      return sizes_.size();
    }

    ptrdiff_t storage_offset() const {
      return storage_offset_;
    }

    // represents that numel() == 0.
    inline bool is_empty() const {
      for (int64_t i = 0; i < dim(); ++i) {
        if (sizes_[i] == 0) {
          return true;
        }
      }
      return false;
    }

    int64_t size(int64_t d) const {
      d = at::maybe_wrap_dim(d, dim(), false);
      return sizes_[d];
    }

    int64_t stride(int64_t d) const {
      d = at::maybe_wrap_dim(d, dim(), false);
      return strides_[d];
    }

    inline at::IntList sizes() {
      return sizes_;
    }

    inline at::IntList strides() {
      return strides_;
    }

    void retain() {
      ++refcount_;
    }

    void release() {
      if(--refcount_ == 0) {
        delete this;
      }
    }
};

inline int64_t* THTensor_getSizePtr(THTensor* tensor) {
  return tensor->sizes_.data();
}

inline int64_t* THTensor_getStridePtr(THTensor* tensor) {
  return tensor->strides_.data();
}

// NB: Non-retaining
inline THStorage* THTensor_getStoragePtr(const THTensor* tensor) {
  return tensor->storage_;
}

#include "generic/THTensorFastGetSet.hpp"
#include "THGenerateAllTypes.h"

inline void THTensor_resizeDim(THTensor* tensor, int64_t ndim) {
  // NB: This is *truly* a resize; calling code (e.g., squeeze)
  // assumes that old values are preserved
  tensor->sizes_.resize(ndim);
  tensor->strides_.resize(ndim);
}

inline void THTensor_setSizesAndStrides(THTensor* tensor, std::vector<int64_t>&& new_size, std::vector<int64_t>&& new_stride) {
  tensor->sizes_ = std::move(new_size);
  tensor->strides_ = std::move(new_stride);
}

inline void THTensor_setSizeAtDim(THTensor* tensor, int dim, int64_t new_size) {
  tensor->sizes_[dim] = new_size;
}

inline void THTensor_setStrideAtDim(THTensor* tensor, int dim, int64_t new_stride) {
  tensor->strides_[dim] = new_stride;
}

inline void THTensor_setStorageOffset(THTensor* tensor, ptrdiff_t storage_offset) {
  tensor->storage_offset_ = storage_offset;
}

// NB: Steals ownership of storage
inline void THTensor_stealAndSetStoragePtr(THTensor* tensor, THStorage* storage) {
  tensor->storage_ = storage;
}

inline bool THTensor_isZeroDim(const THTensor *tensor) {
  return tensor->is_zero_dim_;
}

inline void THTensor_setIsZeroDim(THTensor *tensor, bool is_zero_dim) {
  tensor->is_zero_dim_ = is_zero_dim;
}

TH_API void THTensor_free(THTensor *self);
TH_CPP_API at::optional<std::vector<int64_t>> THTensor_compute_stride(at::IntList oldshape, at::IntList oldstride,
                                                                      at::IntList newshape);
