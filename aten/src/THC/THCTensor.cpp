#include <THC/THCGeneral.h>
#include <THC/THCTensor.hpp>

#include <new>

#include <THC/generic/THCTensor.cpp>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCTensor.cpp>
#include <THC/THCGenerateComplexTypes.h>

#include <THC/generic/THCTensor.cpp>
#include <THC/THCGenerateBoolType.h>

#include <THC/generic/THCTensor.cpp>
#include <THC/THCGenerateBFloat16Type.h>

#include <ATen/native/cuda/Resize.h>

THCTensor *THCTensor_new(THCState *state, caffe2::TypeMeta type_meta) {
  auto scalar_type = at::typeMetaToScalarType(type_meta);
  switch (scalar_type) {
    case at::ScalarType::Byte:
      return THCudaByteTensor_new(state);
    case at::ScalarType::Char:
      return THCudaCharTensor_new(state);
    case at::ScalarType::Short:
      return THCudaShortTensor_new(state);
    case at::ScalarType::Int:
      return THCudaIntTensor_new(state);
    case at::ScalarType::Long:
      return THCudaLongTensor_new(state);
    case at::ScalarType::Half:
      return THCudaHalfTensor_new(state);
    case at::ScalarType::Float:
      return THCudaTensor_new(state);
    case at::ScalarType::Double:
      return THCudaDoubleTensor_new(state);
    case at::ScalarType::Bool:
      return THCudaBoolTensor_new(state);
    case at::ScalarType::BFloat16:
      return THCudaBFloat16Tensor_new(state);
    case at::ScalarType::ComplexFloat:
      return THCudaComplexFloatTensor_new(state);
    case at::ScalarType::ComplexDouble:
      return THCudaComplexDoubleTensor_new(state);
    default:
      AT_ERROR("unexpected ScalarType: ", toString(scalar_type));
  }
}

void THCTensor_resizeAs(THCState *state, THCTensor *self, THCTensor *src) {
  int isSame = 0;
  int d;
  if(self->dim() == src->dim())
  {
    isSame = 1;
    for(d = 0; d < self->dim(); d++)
    {
      if(self->size(d) != src->size(d))
      {
        isSame = 0;
        break;
      }
    }
  }

  if(!isSame)
    THCTensor_resizeNd(state, self, src->dim(), THTensor_getSizePtr(src), NULL);
}

void THCTensor_resizeNd(THCState *state, THCTensor *self, int nDimension, const int64_t *size, const int64_t *stride)
{
  TORCH_CHECK(nDimension >= 0, "resizeNd nDimension must be non-negative");
  at::IntArrayRef sizes(size, nDimension);
  at::optional<at::IntArrayRef> strides;
  if (stride) {
    strides = at::IntArrayRef(stride, nDimension);
  }
  at::native::resize_impl_cuda_(self, sizes, strides, /*device_guard=*/false);
}

void THCTensor_setStorage(THCState *state, THCTensor *self, THCStorage *storage_, ptrdiff_t storageOffset_, at::IntArrayRef size_, at::IntArrayRef stride_)
{
  c10::raw::intrusive_ptr::incref(storage_);
  THTensor_wrap(self).set_(at::Storage(c10::intrusive_ptr<at::StorageImpl>::reclaim(storage_)),
                           storageOffset_, size_, stride_);
}

// NB: It is INVALID to call this on an UndefinedTensor
void THCTensor_retain(THCState *state, THCTensor *self) {
  c10::raw::intrusive_ptr::incref(self);
}

void THCTensor_free(THCState *state, THCTensor *self) {
  THTensor_free(self);
}

int THCTensor_getDevice(THCState* state, const THCTensor* tensor) {
  if (!THTensor_getStoragePtr(tensor)) return -1;
  return THCStorage_getDevice(state, THTensor_getStoragePtr(tensor));
}
