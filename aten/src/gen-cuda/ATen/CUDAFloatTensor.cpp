// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/Config.h"
#include "ATen/CUDAFloatTensor.h"
#include "ATen/CUDAFloatStorage.h"
#include "ATen/Scalar.h"
#include "ATen/Half.h"

#if AT_CUDA_ENABLED()
#include <ATen/cuda/CUDAHalf.cuh>
#endif

namespace at {

CUDAFloatTensor::CUDAFloatTensor(Context* context)
: CUDAFloatTensor(context,THCudaTensor_new(context->thc_state)) {}

CUDAFloatTensor::CUDAFloatTensor(Context* context, THCudaTensor * tensor)
: TensorImpl(&context->getType(Backend::CUDA,ScalarType::Float)),
  tensor(tensor),
  context(context) {}
CUDAFloatTensor::~CUDAFloatTensor() {
  THCudaTensor_free(context->thc_state,  tensor);
}

const char * CUDAFloatTensor::toString() const {
  return "CUDAFloatTensor";
}

IntList CUDAFloatTensor::sizes() const {
  int64_t d = tensor->nDimension;
  if (d != 0) {
    // note: this will return "{}" for a scalar because dim() will return 0 in that case.
    return IntList(reinterpret_cast<int64_t*>(tensor->size),dim());
  } else {
    return IntList(kEmptySizes);
  }
}

int64_t CUDAFloatTensor::dim() const {
  if(isScalar())
    return 0;
  int64_t d = tensor->nDimension;
  // See Note [Empty versus 0-dim tensors]
  if (d != 0)
    return d;
  return kEmptySizes.size();
}

const char * CUDAFloatTensor::typeString() {
  return "CUDAFloatType";
}
void * CUDAFloatTensor::unsafeGetTH(bool retain) {
  if (retain)
      THCudaTensor_retain(context->thc_state,  tensor);
  return tensor;
}

// included as 'TensorDenseOrSparse' in TensorDerived.cpp

IntList CUDAFloatTensor::strides() const {
  int64_t d = tensor->nDimension;
  if (d != 0) {
    return IntList(reinterpret_cast<int64_t*>(tensor->stride),dim());
  } else {
    return IntList(kEmptyStrides);
  }
}
Scalar CUDAFloatTensor::localScalar() {
  int64_t numel = THCudaTensor_nElement(context->thc_state, tensor);
  AT_ASSERT(numel == 1,"localScalar() called on Tensor with %" PRId64 " elements",numel);
  return Scalar((THCudaStorage_get(context->thc_state, tensor->storage, tensor->storageOffset)));
}
std::unique_ptr<Storage> CUDAFloatTensor::storage() {
  auto storage = THCudaTensor_storage(context->thc_state, tensor);
  THCudaStorage_retain(context->thc_state, storage);
  return std::unique_ptr<Storage>(new CUDAFloatStorage(&type().get_context(), storage));
}


}
