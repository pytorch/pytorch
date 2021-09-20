#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensor.cpp"
#else

#include <ATen/InferSize.h>
#include <ATen/NativeFunctions.h>

/**** access methods ****/
THCStorage *THCTensor_(storage)(THCState *state, const THCTensor *self)
{
  return THTensor_getStoragePtr(self);
}

/**** creation methods ****/

/* Empty init */
THCTensor *THCTensor_(new)(THCState *state)
{
  return c10::make_intrusive<at::TensorImpl, at::UndefinedTensorImpl>(
             c10::intrusive_ptr<at::StorageImpl>::reclaim(
                 THCStorage_(new)(state)),
             at::DispatchKey::CUDA,
             caffe2::TypeMeta::Make<scalar_t>())
      .release();
}


THCTensor *THCTensor_(newWithStorage1d)(THCState *state, THCStorage *storage, ptrdiff_t storageOffset,
                               int64_t size0, int64_t stride0)
{
  c10::raw::intrusive_ptr::incref(storage);
  THTensor* self = c10::make_intrusive<at::TensorImpl, at::UndefinedTensorImpl>(
                       c10::intrusive_ptr<at::StorageImpl>::reclaim(storage),
                       at::DispatchKey::CUDA,
                       caffe2::TypeMeta::Make<scalar_t>())
                       .release();
  THCTensor_setStorage(state, self, storage, storageOffset, {size0}, {stride0});

  return self;
}

// void THCTensor_(setStorage)(THCState *state, THCTensor *self, THCStorage *storage_, ptrdiff_t storageOffset_, at::IntArrayRef size_, at::IntArrayRef stride_) {
//   THCTensor_setStorage(state, self, storage_, storageOffset_, size_, stride_);
// }
void THCTensor_(free)(THCState *state, THCTensor *self)
{
  THCTensor_free(state, self);
}
#endif
