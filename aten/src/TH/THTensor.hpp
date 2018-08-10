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

    // Note: storage->size() may be greater than the recorded size
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

    inline int64_t dim() const {
      return sizes_.size();
    }

    at::ScalarType scalar_type() const {
      return storage_->scalar_type();
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
};

inline int64_t* THTensor_getSizePtr(THTensor* tensor) {
  return tensor->sizes_.data();
}

inline int64_t* THTensor_getStridePtr(THTensor* tensor) {
  return tensor->strides_.data();
}

// NB: Non-retaining
inline THStorage* THTensor_getStoragePtr(const THTensor* tensor) {
  // Within PyTorch, the invariant is that storage_ is always
  // initialized; we never have tensors that don't have any storage.
  // However, for Caffe2, this is not true, because they have permitted
  // tensors to be allocated without specifying what scalar type
  // they should be, only to be filled when GetMutableData is called
  // for the first time (providing the necessary type).  It is an ERROR to
  // invoke any PyTorch operations on such a half-constructed storage,
  // and this check tests for that case.
  AT_CHECK(tensor->storage_, "Cannot use PyTorch operations on a half-constructed "
           "tensor.  If this tensor came from Caffe2, please call GetMutableData on "
           "it first; otherwise, this is a bug, please report it.");
  return tensor->storage_;
}

inline bool THTensor_isZeroDim(const THTensor *tensor) {
  return tensor->is_zero_dim_;
}

inline void THTensor_setIsZeroDim(THTensor *tensor, bool is_zero_dim) {
  tensor->is_zero_dim_ = is_zero_dim;
}

inline void THTensor_maybe_zero_dim(THTensor *tensor, bool condition_when_zero_dim) {
  bool is_zero_dim = (condition_when_zero_dim && tensor->sizes().size() == 1 && tensor->size(0) == 1) || tensor->dim() == 0;
  THTensor_setIsZeroDim(tensor, is_zero_dim);
}

// [NOTE: nDimension vs nDimensionLegacyNoScalars vs nDimensionLegacyAll]
// nDimension                 corresponds to the "true" ATen dimension. TODO: implement.
// nDimensionLegacyNoScalars  correpsonds to the ATen dimension, except scalars are viewed as 1-dimensional tensors.
// nDimensionLegacyAll        corresponds to the ATen dimension, except scalars are viewed as 1-dimensional tensors
//                            and tensors with a dimension of size zero are collapsed to 0-dimensional tensors.
//
// Eventually, everything should go through nDimension or tensor->dim().
inline int THTensor_nDimension(const THTensor* tensor) {
  return tensor->dim();
}

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

inline int64_t THTensor_strideLegacyNoScalars(const THTensor *self, int dim) {
  THArgCheck((dim >= 0) && (dim < THTensor_nDimensionLegacyNoScalars(self)), 2, "dimension %d out of range of %dD tensor",
      dim+TH_INDEX_BASE, THTensor_nDimensionLegacyNoScalars(self));
  return THTensor_isZeroDim(self) ? 1 : self->stride(dim);
}

inline int64_t THTensor_sizeLegacyNoScalars(const THTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < THTensor_nDimensionLegacyNoScalars(self)), 2, "dimension %d out of range of %dD tensor",
      dim+TH_INDEX_BASE, THTensor_nDimensionLegacyNoScalars(self));
  return THTensor_isZeroDim(self) ? 1 : self->size(dim);
}

#include "generic/THTensorFastGetSet.hpp"
#include "THGenerateAllTypes.h"

inline std::vector<int64_t> THTensor_sizesLegacyNoScalars(const THTensor *self) {
  if (self->dim() == 0) {
    return {1};
  } else {
    return self->sizes().vec();
  }
}

inline std::vector<int64_t> THTensor_stridesLegacyNoScalars(const THTensor *self) {
  if (self->dim() == 0) {
    return {1};
  } else {
    return self->strides().vec();
  }
}

inline void THTensor_resizeDim(THTensor* tensor, int64_t ndim) {
  // NB: This is *truly* a resize; calling code (e.g., squeeze)
  // assumes that old values are preserved
  tensor->is_zero_dim_ = bool(ndim == 0);
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
  // Caffe2 might have tensors whose storages are null, but we
  // don't allow it in PyTorch.
  AT_ASSERT(storage);
  tensor->storage_ = storage;
}

TH_API void THTensor_free(THTensor *self);
TH_API void THTensor_setStorageNd(THTensor *self, THStorage *storage, ptrdiff_t storageOffset, int nDimension, const int64_t *size, const int64_t *stride);
TH_API void THTensor_resizeNd(THTensor *self, int nDimension, const int64_t *size, const int64_t *stride);

TH_CPP_API void THTensor_resize(THTensor *self, at::IntList size, at::IntList stride);
TH_CPP_API void THTensor_setStorage(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_, at::IntList size_, at::IntList stride_);
TH_CPP_API at::optional<std::vector<int64_t>> THTensor_compute_stride(at::IntList oldshape, at::IntList oldstride,
                                                                      at::IntList newshape);

#include "generic/THTensor.hpp"
#include "THGenerateAllTypes.h"

#include "generic/THTensor.hpp"
#include "THGenerateHalfType.h"
