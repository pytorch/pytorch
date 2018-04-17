// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/Config.h"
#include "ATen/CUDAByteTensor.h"
#include "ATen/CUDAByteStorage.h"
#include "ATen/Scalar.h"
#include "ATen/Half.h"

#if AT_CUDA_ENABLED()
#include <ATen/cuda/CUDAHalf.cuh>
#endif

namespace at {

CUDAByteTensor::CUDAByteTensor(Context* context)
: CUDAByteTensor(context,THCudaByteTensor_new(context->thc_state)) {}

CUDAByteTensor::CUDAByteTensor(Context* context, THCudaByteTensor * tensor)
: TensorImpl(&context->getType(Backend::CUDA,ScalarType::Byte)),
  tensor(tensor),
  context(context) {}
CUDAByteTensor::~CUDAByteTensor() {
  THCudaByteTensor_free(context->thc_state,  tensor);
}

const char * CUDAByteTensor::toString() const {
  return "CUDAByteTensor";
}

IntList CUDAByteTensor::sizes() const {
  int64_t d = tensor->nDimension;
  if (d != 0) {
    // note: this will return "{}" for a scalar because dim() will return 0 in that case.
    return IntList(reinterpret_cast<int64_t*>(tensor->size),dim());
  } else {
    return IntList(kEmptySizes);
  }
}

int64_t CUDAByteTensor::dim() const {
  if(isScalar())
    return 0;
  int64_t d = tensor->nDimension;
  // See Note [Empty versus 0-dim tensors]
  if (d != 0)
    return d;
  return kEmptySizes.size();
}

const char * CUDAByteTensor::typeString() {
  return "CUDAByteType";
}
void * CUDAByteTensor::unsafeGetTH(bool retain) {
  if (retain)
      THCudaByteTensor_retain(context->thc_state,  tensor);
  return tensor;
}

// included as 'TensorDenseOrSparse' in TensorDerived.cpp

IntList CUDAByteTensor::strides() const {
  int64_t d = tensor->nDimension;
  if (d != 0) {
    return IntList(reinterpret_cast<int64_t*>(tensor->stride),dim());
  } else {
    return IntList(kEmptyStrides);
  }
}
Scalar CUDAByteTensor::localScalar() {
  int64_t numel = THCudaByteTensor_nElement(context->thc_state, tensor);
  AT_ASSERT(numel == 1,"localScalar() called on Tensor with %" PRId64 " elements",numel);
  return Scalar((THCudaByteStorage_get(context->thc_state, tensor->storage, tensor->storageOffset)));
}
std::unique_ptr<Storage> CUDAByteTensor::storage() {
  auto storage = THCudaByteTensor_storage(context->thc_state, tensor);
  THCudaByteStorage_retain(context->thc_state, storage);
  return std::unique_ptr<Storage>(new CUDAByteStorage(&type().get_context(), storage));
}


}
