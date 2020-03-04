#include <TH/THTensor.hpp>

#include <TH/generic/THTensor.cpp>
#include <TH/THGenerateAllTypes.h>

#include <TH/generic/THTensor.cpp>
#include <TH/THGenerateHalfType.h>

#include <TH/generic/THTensor.cpp>
#include <TH/THGenerateBoolType.h>

#include <TH/generic/THTensor.cpp>
#include <TH/THGenerateBFloat16Type.h>

#include <ATen/native/Resize.h>
#include <ATen/native/TensorShape.h>
#include <ATen/TensorUtils.h>

#include <numeric>

// NB: This is NOT valid on UndefinedTensorImpl
void THTensor_free(THTensor *self)
{
  if (!self) return;
  c10::raw::intrusive_ptr::decref(self);
}

void THTensor_setStorage(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_, at::IntArrayRef size_, at::IntArrayRef stride_) {
  c10::raw::intrusive_ptr::incref(storage_);
  at::Storage at_storage(c10::intrusive_ptr<at::StorageImpl>::reclaim(storage_));
  auto at_self = THTensor_wrap(self);
  at::native::set_cpu_(at_self, at_storage, storageOffset_, size_, stride_);
}

void THTensor_setStorageNd(THTensor *self, THStorage *storage, ptrdiff_t storageOffset, int nDimension, const int64_t *size, const int64_t *stride) {
  at::IntArrayRef at_stride;
  if (stride) {
    at_stride = at::IntArrayRef(stride, nDimension);
  }
  THTensor_setStorage(self, storage, storageOffset, at::IntArrayRef(size, nDimension), at_stride);
}

void THTensor_resize(THTensor *self, at::IntArrayRef size, at::IntArrayRef stride)
{
  if (stride.data()) {
    THArgCheck(stride.size() == size.size(), 3, "invalid stride");
  }

#ifdef DEBUG
  THAssert(size.size() <= INT_MAX);
#endif
  THTensor_resizeNd(self, size.size(), size.data(), stride.data());
}

void THTensor_resizeNd(THTensor *self, int nDimension, const int64_t *size, const int64_t *stride)
{
  TORCH_CHECK(nDimension >= 0, "resizeNd nDimension must be non-negative");
  at::IntArrayRef sizes(size, nDimension);
  at::optional<at::IntArrayRef> strides;
  if (stride) {
    strides = at::IntArrayRef(stride, nDimension);
  }
  at::native::resize_impl_cpu_(self, sizes, strides);
}
