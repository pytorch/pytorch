// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/Config.h"
#include "ATen/CUDADoubleTensor.h"
#include "ATen/CUDADoubleStorage.h"
#include "ATen/Scalar.h"
#include "ATen/Half.h"

#if AT_CUDA_ENABLED()
#include <ATen/cuda/CUDAHalf.cuh>
#endif

namespace at {

CUDADoubleTensor::CUDADoubleTensor(Context* context)
: CUDADoubleTensor(context,THCudaDoubleTensor_new(context->thc_state)) {}

CUDADoubleTensor::CUDADoubleTensor(Context* context, THCudaDoubleTensor * tensor)
: TensorImpl(&context->getType(Backend::CUDA,ScalarType::Double)),
  tensor(tensor),
  context(context) {}
CUDADoubleTensor::~CUDADoubleTensor() {
  THCudaDoubleTensor_free(context->thc_state,  tensor);
}

const char * CUDADoubleTensor::toString() const {
  return "CUDADoubleTensor";
}

IntList CUDADoubleTensor::sizes() const {
  int64_t d = tensor->nDimension;
  if (d != 0) {
    // note: this will return "{}" for a scalar because dim() will return 0 in that case.
    return IntList(reinterpret_cast<int64_t*>(tensor->size),dim());
  } else {
    return IntList(kEmptySizes);
  }
}

int64_t CUDADoubleTensor::dim() const {
  if(isScalar())
    return 0;
  int64_t d = tensor->nDimension;
  // See Note [Empty versus 0-dim tensors]
  if (d != 0)
    return d;
  return kEmptySizes.size();
}

const char * CUDADoubleTensor::typeString() {
  return "CUDADoubleType";
}
void * CUDADoubleTensor::unsafeGetTH(bool retain) {
  if (retain)
      THCudaDoubleTensor_retain(context->thc_state,  tensor);
  return tensor;
}

// included as 'TensorDenseOrSparse' in TensorDerived.cpp

IntList CUDADoubleTensor::strides() const {
  int64_t d = tensor->nDimension;
  if (d != 0) {
    return IntList(reinterpret_cast<int64_t*>(tensor->stride),dim());
  } else {
    return IntList(kEmptyStrides);
  }
}
Scalar CUDADoubleTensor::localScalar() {
  int64_t numel = THCudaDoubleTensor_nElement(context->thc_state, tensor);
  AT_ASSERT(numel == 1,"localScalar() called on Tensor with %" PRId64 " elements",numel);
  return Scalar((THCudaDoubleStorage_get(context->thc_state, tensor->storage, tensor->storageOffset)));
}
std::unique_ptr<Storage> CUDADoubleTensor::storage() {
  auto storage = THCudaDoubleTensor_storage(context->thc_state, tensor);
  THCudaDoubleStorage_retain(context->thc_state, storage);
  return std::unique_ptr<Storage>(new CUDADoubleStorage(&type().get_context(), storage));
}


}
