// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/Config.h"
#include "ATen/CUDAIntTensor.h"
#include "ATen/CUDAIntStorage.h"
#include "ATen/Scalar.h"
#include "ATen/Half.h"

#if AT_CUDA_ENABLED()
#include <ATen/cuda/CUDAHalf.cuh>
#endif

namespace at {

CUDAIntTensor::CUDAIntTensor(Context* context)
: CUDAIntTensor(context,THCudaIntTensor_new(context->thc_state)) {}

CUDAIntTensor::CUDAIntTensor(Context* context, THCudaIntTensor * tensor)
: TensorImpl(&context->getType(Backend::CUDA,ScalarType::Int)),
  tensor(tensor),
  context(context) {}
CUDAIntTensor::~CUDAIntTensor() {
  THCudaIntTensor_free(context->thc_state,  tensor);
}

const char * CUDAIntTensor::toString() const {
  return "CUDAIntTensor";
}

IntList CUDAIntTensor::sizes() const {
  int64_t d = tensor->nDimension;
  if (d != 0) {
    // note: this will return "{}" for a scalar because dim() will return 0 in that case.
    return IntList(reinterpret_cast<int64_t*>(tensor->size),dim());
  } else {
    return IntList(kEmptySizes);
  }
}

int64_t CUDAIntTensor::dim() const {
  if(isScalar())
    return 0;
  int64_t d = tensor->nDimension;
  // See Note [Empty versus 0-dim tensors]
  if (d != 0)
    return d;
  return kEmptySizes.size();
}

const char * CUDAIntTensor::typeString() {
  return "CUDAIntType";
}
void * CUDAIntTensor::unsafeGetTH(bool retain) {
  if (retain)
      THCudaIntTensor_retain(context->thc_state,  tensor);
  return tensor;
}

// included as 'TensorDenseOrSparse' in TensorDerived.cpp

IntList CUDAIntTensor::strides() const {
  int64_t d = tensor->nDimension;
  if (d != 0) {
    return IntList(reinterpret_cast<int64_t*>(tensor->stride),dim());
  } else {
    return IntList(kEmptyStrides);
  }
}
Scalar CUDAIntTensor::localScalar() {
  int64_t numel = THCudaIntTensor_nElement(context->thc_state, tensor);
  AT_ASSERT(numel == 1,"localScalar() called on Tensor with %" PRId64 " elements",numel);
  return Scalar((THCudaIntStorage_get(context->thc_state, tensor->storage, tensor->storageOffset)));
}
std::unique_ptr<Storage> CUDAIntTensor::storage() {
  auto storage = THCudaIntTensor_storage(context->thc_state, tensor);
  THCudaIntStorage_retain(context->thc_state, storage);
  return std::unique_ptr<Storage>(new CUDAIntStorage(&type().get_context(), storage));
}


}
