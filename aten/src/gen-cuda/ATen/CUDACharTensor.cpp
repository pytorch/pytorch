// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/Config.h"
#include "ATen/CUDACharTensor.h"
#include "ATen/CUDACharStorage.h"
#include "ATen/Scalar.h"
#include "ATen/Half.h"

#if AT_CUDA_ENABLED()
#include <ATen/cuda/CUDAHalf.cuh>
#endif

namespace at {

CUDACharTensor::CUDACharTensor(Context* context)
: CUDACharTensor(context,THCudaCharTensor_new(context->thc_state)) {}

CUDACharTensor::CUDACharTensor(Context* context, THCudaCharTensor * tensor)
: TensorImpl(&context->getType(Backend::CUDA,ScalarType::Char)),
  tensor(tensor),
  context(context) {}
CUDACharTensor::~CUDACharTensor() {
  THCudaCharTensor_free(context->thc_state,  tensor);
}

const char * CUDACharTensor::toString() const {
  return "CUDACharTensor";
}

IntList CUDACharTensor::sizes() const {
  int64_t d = tensor->nDimension;
  if (d != 0) {
    // note: this will return "{}" for a scalar because dim() will return 0 in that case.
    return IntList(reinterpret_cast<int64_t*>(tensor->size),dim());
  } else {
    return IntList(kEmptySizes);
  }
}

int64_t CUDACharTensor::dim() const {
  if(isScalar())
    return 0;
  int64_t d = tensor->nDimension;
  // See Note [Empty versus 0-dim tensors]
  if (d != 0)
    return d;
  return kEmptySizes.size();
}

const char * CUDACharTensor::typeString() {
  return "CUDACharType";
}
void * CUDACharTensor::unsafeGetTH(bool retain) {
  if (retain)
      THCudaCharTensor_retain(context->thc_state,  tensor);
  return tensor;
}

// included as 'TensorDenseOrSparse' in TensorDerived.cpp

IntList CUDACharTensor::strides() const {
  int64_t d = tensor->nDimension;
  if (d != 0) {
    return IntList(reinterpret_cast<int64_t*>(tensor->stride),dim());
  } else {
    return IntList(kEmptyStrides);
  }
}
Scalar CUDACharTensor::localScalar() {
  int64_t numel = THCudaCharTensor_nElement(context->thc_state, tensor);
  AT_ASSERT(numel == 1,"localScalar() called on Tensor with %" PRId64 " elements",numel);
  return Scalar((THCudaCharStorage_get(context->thc_state, tensor->storage, tensor->storageOffset)));
}
std::unique_ptr<Storage> CUDACharTensor::storage() {
  auto storage = THCudaCharTensor_storage(context->thc_state, tensor);
  THCudaCharStorage_retain(context->thc_state, storage);
  return std::unique_ptr<Storage>(new CUDACharStorage(&type().get_context(), storage));
}


}
