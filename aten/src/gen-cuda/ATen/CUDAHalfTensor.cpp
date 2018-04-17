// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/Config.h"
#include "ATen/CUDAHalfTensor.h"
#include "ATen/CUDAHalfStorage.h"
#include "ATen/Scalar.h"
#include "ATen/Half.h"

#if AT_CUDA_ENABLED()
#include <ATen/cuda/CUDAHalf.cuh>
#endif

namespace at {

CUDAHalfTensor::CUDAHalfTensor(Context* context)
: CUDAHalfTensor(context,THCudaHalfTensor_new(context->thc_state)) {}

CUDAHalfTensor::CUDAHalfTensor(Context* context, THCudaHalfTensor * tensor)
: TensorImpl(&context->getType(Backend::CUDA,ScalarType::Half)),
  tensor(tensor),
  context(context) {}
CUDAHalfTensor::~CUDAHalfTensor() {
  THCudaHalfTensor_free(context->thc_state,  tensor);
}

const char * CUDAHalfTensor::toString() const {
  return "CUDAHalfTensor";
}

IntList CUDAHalfTensor::sizes() const {
  int64_t d = tensor->nDimension;
  if (d != 0) {
    // note: this will return "{}" for a scalar because dim() will return 0 in that case.
    return IntList(reinterpret_cast<int64_t*>(tensor->size),dim());
  } else {
    return IntList(kEmptySizes);
  }
}

int64_t CUDAHalfTensor::dim() const {
  if(isScalar())
    return 0;
  int64_t d = tensor->nDimension;
  // See Note [Empty versus 0-dim tensors]
  if (d != 0)
    return d;
  return kEmptySizes.size();
}

const char * CUDAHalfTensor::typeString() {
  return "CUDAHalfType";
}
void * CUDAHalfTensor::unsafeGetTH(bool retain) {
  if (retain)
      THCudaHalfTensor_retain(context->thc_state,  tensor);
  return tensor;
}

// included as 'TensorDenseOrSparse' in TensorDerived.cpp

IntList CUDAHalfTensor::strides() const {
  int64_t d = tensor->nDimension;
  if (d != 0) {
    return IntList(reinterpret_cast<int64_t*>(tensor->stride),dim());
  } else {
    return IntList(kEmptyStrides);
  }
}
Scalar CUDAHalfTensor::localScalar() {
  int64_t numel = THCudaHalfTensor_nElement(context->thc_state, tensor);
  AT_ASSERT(numel == 1,"localScalar() called on Tensor with %" PRId64 " elements",numel);
  return Scalar(HalfFix<Half,__half>(THCudaHalfStorage_get(context->thc_state, tensor->storage, tensor->storageOffset)));
}
std::unique_ptr<Storage> CUDAHalfTensor::storage() {
  auto storage = THCudaHalfTensor_storage(context->thc_state, tensor);
  THCudaHalfStorage_retain(context->thc_state, storage);
  return std::unique_ptr<Storage>(new CUDAHalfStorage(&type().get_context(), storage));
}


}
