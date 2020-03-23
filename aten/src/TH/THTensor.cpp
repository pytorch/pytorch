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
  THTensor_wrap(self).set_(at::Storage(c10::intrusive_ptr<at::StorageImpl>::reclaim(storage_)), storageOffset_, size_, stride_);
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

// NB: Steals ownership of storage
void THTensor_stealAndSetStoragePtr(THTensor* tensor, THStorage* storage) {
  // Caffe2 might have tensors whose storages are null, but we
  // don't allow it in PyTorch.
  AT_ASSERT(storage);
  // Caffe2 also has uninitialized dtype states, which we disallow here
  AT_ASSERT(tensor->storage().dtype() == storage->dtype());

  // We used to allow this, but this breaks device caching.
  // Let's put an actual error message for this one.
  TORCH_CHECK(tensor->storage().device() == storage->device(),
            "Attempted to set the storage of a tensor on device \"", tensor->storage().device(),
             "\" to a storage on different device \"", storage->device(),
            "\".  This is no longer allowed; the devices must match.");
  tensor->set_storage(at::Storage(c10::intrusive_ptr<THStorage>::reclaim(storage)));
}
