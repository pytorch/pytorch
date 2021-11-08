#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THTensor.cpp"
#else

#include <ATen/InferSize.h>
#include <ATen/NativeFunctions.h>
#include <new>
#include <ATen/NamedTensorUtils.h>
#include <ATen/MemoryOverlap.h>

/**** creation methods ****/

THTensor *THTensor_(newWithStorage1d)(THStorage *storage, ptrdiff_t storageOffset,
                               int64_t size0, int64_t stride0)
{
  c10::raw::intrusive_ptr::incref(storage);
  THTensor* self = c10::make_intrusive<at::TensorImpl, at::UndefinedTensorImpl>(
                       c10::intrusive_ptr<at::StorageImpl>::reclaim(storage),
                       at::DispatchKey::CPU,
#ifdef THQUANTIZED
                       caffe2::TypeMeta::Make<quantized_t>()
#else
                       caffe2::TypeMeta::Make<scalar_t>()
#endif
                           )
                       .release();
  THTensor_setStorage(self, storage, storageOffset,  {size0}, {stride0});

  return self;
}
#endif
