#pragma once

// STOP!!! Thinking of including this header directly?  Please
// read Note [TH abstraction violation]

#include "THTensor.h"
#include "THStorageFunctions.hpp"

#include <atomic>
#include <ATen/ATen.h>

struct THTensor
{
public:
    THTensor(THStorage* storage)
      : refcount_(1)
      , storage_(storage)
      , storage_offset_(0)
      , sizes_{0}
      , strides_{1}
      , numel_(0)
      , is_zero_dim_(false)
      {}

    ~THTensor() {
      if (storage_) {
        THStorage_free(storage_);
      }
    }

private:
    std::atomic<int> refcount_;

    // Note: storage->size may be greater than the recorded size
    // of a tensor
    THStorage *storage_;
    ptrdiff_t storage_offset_;

    std::vector<int64_t> sizes_;
    std::vector<int64_t> strides_;

    int64_t numel_;

public:
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

    inline int64_t dim() const {
      return sizes_.size();
    }

    inline int64_t numel() const {
      return numel_;
    }

    ptrdiff_t storage_offset() const {
      return storage_offset_;
    }

    inline bool is_empty() const {
      return numel_ == 0;
    }

    int64_t size(int64_t d) const {
      d = at::maybe_wrap_dim(d, dim(), false);
      return sizes_[d];
    }

    int64_t stride(int64_t d) const {
      d = at::maybe_wrap_dim(d, dim(), false);
      return strides_[d];
    }

    // WARNING: This function does not check if the requested
    // sizes/strides are in bounds for the storage is allocated;
    // this is the responsibility of the caller
    void as_strided_(at::IntList sizes, at::IntList strides) {
      AT_ASSERT(sizes.size() == strides.size());
      sizes_ = sizes.vec();
      strides_ = strides.vec();
      for (auto s : sizes_) {
        numel_ *= s;
        if (numel_ == 0) break;
      }
    }

    inline at::IntList sizes() const {
      return sizes_;
    }

    inline at::IntList strides() const {
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

    friend void THTensor_resizeDim(THTensor* tensor, int64_t ndim);
    friend void THTensor_setSizeAtDim(THTensor* tensor, int dim, int64_t new_size);
    friend void THTensor_setStrideAtDim(THTensor* tensor, int dim, int64_t new_stride);
    friend void THTensor_setStorageOffset(THTensor* tensor, ptrdiff_t storage_offset);
    friend THStorage* THTensor_getStoragePtr(const THTensor* tensor);
    friend void THTensor_stealAndSetStoragePtr(THTensor* tensor, THStorage* storage);
};

inline const int64_t* THTensor_getSizePtr(THTensor* tensor) {
  return tensor->sizes().data();
}

inline const int64_t* THTensor_getStridePtr(THTensor* tensor) {
  return tensor->strides().data();
}

// NB: Non-retaining
inline THStorage* THTensor_getStoragePtr(const THTensor* tensor) {
  return tensor->storage_;
}

#include "generic/THTensorFastGetSet.hpp"
#include "THGenerateAllTypes.h"

// NB: This is *truly* a resize; calling code (e.g., squeeze)
// assumes that old values are preserved
// TODO: Get rid of this function; it's a bit inefficient from
// a calculating cached numels perspective.
inline void THTensor_resizeDim(THTensor* tensor, int64_t ndim) {
  AT_ASSERT(ndim >= 0);
  // Default the new sizes and strides to 1, because (a) it gives a size
  // stride which is exactly equivalent to the old sizes/strides
  // and (b) it doesn't change numel.
  tensor->sizes_.resize(ndim, 1);
  tensor->strides_.resize(ndim, 1);
  tensor->numel_ = 1;
  for (int64_t i = 0; i < ndim; i++) {
    tensor->numel_ *= tensor->sizes_[i];
    if (tensor->numel_ == 0) break;
  }
}

inline void THTensor_setSizeAtDim(THTensor* tensor, int dim, int64_t new_size) {
  if (tensor->sizes_[dim] == 0) {
    // Gotta fully recompute
    // TODO: This can lead to bad asymptotics O(dim^2) if you call setSizeAtDim
    // on a loop for a tensor with many zero-size dimensions.
    tensor->numel_ = 1;
    for (int64_t i = 0; i < static_cast<int64_t>(tensor->sizes_.size()); i++) {
      if (i == dim) {
        tensor->numel_ *= new_size;
      } else {
        tensor->numel_ *= tensor->sizes_[i];
      }
      if (tensor->numel_ == 0) break;
    }
  } else {
    tensor->numel_ /= tensor->sizes_[dim];
    tensor->numel_ *= new_size;
  }
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

// [NOTE: nDimension vs nDimensionLegacyNoScalars vs nDimensionLegacyAll]
// nDimension                 corresponds to the "true" ATen dimension. TODO: implement.
// nDimensionLegacyNoScalars  correpsonds to the ATen dimension, except scalars are viewed as 1-dimensional tensors.
// nDimensionLegacyAll        corresponds to the ATen dimension, except scalars are viewed as 1-dimensional tensors
//                            and tensors with a dimension of size zero are collapsed to 0-dimensional tensors.
//
// Eventually, everything should go through nDimension or tensor->dim().
inline int THTensor_nDimensionLegacyNoScalars(const THTensor* tensor) {
  if (THTensor_isZeroDim(tensor)) {
    return 1;
  } else {
    return tensor->dim();  
  }
}

inline int THTensor_nDimensionLegacyAll(const THTensor* tensor) {
  if (tensor->is_empty()) {
    return 0;  
  } else if (THTensor_isZeroDim(tensor)) {
    return 1;
  } else {
    return tensor->dim();  
  }
}

TH_API void THTensor_free(THTensor *self);
TH_CPP_API at::optional<std::vector<int64_t>> THTensor_compute_stride(at::IntList oldshape, at::IntList oldstride,
                                                                      at::IntList newshape);
