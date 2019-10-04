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

inline const int64_t* THTensor_getSizePtr(THTensor* tensor) {
  return tensor->sizes().data();
}

inline const int64_t* THTensor_getStridePtr(THTensor* tensor) {
  return tensor->strides().data();
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
  TORCH_CHECK(tensor->storage(), "Cannot use PyTorch operations on a half-constructed "
           "tensor.  If this tensor came from Caffe2, please call GetMutableData on "
           "it first; otherwise, this is a bug, please report it.");
  return tensor->storage().unsafeGetStorageImpl();
}

inline void THTensor_maybe_zero_dim(THTensor *tensor, bool condition_when_zero_dim) {
  bool set_zero_dim = condition_when_zero_dim && tensor->sizes().size() == 1 && tensor->size(0) == 1;
  if (set_zero_dim) {
    tensor->set_sizes_and_strides({}, {});
  }
}

// [NOTE: nDimension vs nDimensionLegacyNoScalars vs nDimensionLegacyAll]
// nDimension                 corresponds to the "true" ATen dimension.
// nDimensionLegacyNoScalars  correpsonds to the ATen dimension, except scalars are viewed as 1-dimensional tensors.
// nDimensionLegacyAll        corresponds to the ATen dimension, except scalars are viewed as 1-dimensional tensors
//                            and tensors with a dimension of size zero are collapsed to 0-dimensional tensors.
//
// Eventually, everything should go through nDimension or tensor->dim().
inline int THTensor_nDimension(const THTensor* tensor) {
  return tensor->dim();
}

inline int THTensor_nDimensionLegacyNoScalars(const THTensor* tensor) {
  if (tensor->dim() == 0) {
    return 1;
  } else {
    return tensor->dim();
  }
}

inline int THTensor_nDimensionLegacyAll(const THTensor* tensor) {
  if (tensor->is_empty()) {
    return 0;
  } else if (tensor->dim() == 0) {
    return 1;
  } else {
    return tensor->dim();
  }
}

inline int64_t THTensor_strideLegacyNoScalars(const THTensor *self, int dim) {
  THArgCheck((dim >= 0) && (dim < THTensor_nDimensionLegacyNoScalars(self)), 2, "dimension %d out of range of %dD tensor",
      dim, THTensor_nDimensionLegacyNoScalars(self));
  return self->dim() == 0 ? 1 : self->stride(dim);
}

inline int64_t THTensor_sizeLegacyNoScalars(const THTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < THTensor_nDimensionLegacyNoScalars(self)), 2, "dimension %d out of range of %dD tensor",
      dim, THTensor_nDimensionLegacyNoScalars(self));
  return self->dim() == 0 ? 1 : self->size(dim);
}

#include <TH/generic/THTensorFastGetSet.hpp>
#include <TH/THGenerateAllTypes.h>

#include <TH/generic/THTensorFastGetSet.hpp>
#include <TH/THGenerateBFloat16Type.h>

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

// NB: Steals ownership of storage
TH_API void THTensor_stealAndSetStoragePtr(THTensor* tensor, THStorage* storage);

TH_API void THTensor_free(THTensor *self);
TH_API void THTensor_setStorageNd(THTensor *self, THStorage *storage, ptrdiff_t storageOffset, int nDimension, const int64_t *size, const int64_t *stride);
TH_API void THTensor_resizeNd(THTensor *self, int nDimension, const int64_t *size, const int64_t *stride);

TH_CPP_API void THTensor_resize(THTensor *self, at::IntArrayRef size, at::IntArrayRef stride);
TH_CPP_API void THTensor_setStorage(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_, at::IntArrayRef size_, at::IntArrayRef stride_);
TH_CPP_API c10::optional<std::vector<int64_t>> THTensor_compute_stride(
    at::IntArrayRef oldshape,
    at::IntArrayRef oldstride,
    at::IntArrayRef newshape);

#include <TH/generic/THTensor.hpp>
#include <TH/THGenerateAllTypes.h>

#include <TH/generic/THTensor.hpp>
#include <TH/THGenerateHalfType.h>

#include <TH/generic/THTensor.hpp>
#include <TH/THGenerateBoolType.h>

#include <TH/generic/THTensor.hpp>
#include <TH/THGenerateBFloat16Type.h>
