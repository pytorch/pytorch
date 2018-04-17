// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/Config.h"
#include "ATen/CUDALongTensor.h"
#include "ATen/CUDALongStorage.h"
#include "ATen/Scalar.h"
#include "ATen/Half.h"

#if AT_CUDA_ENABLED()
#include <ATen/cuda/CUDAHalf.cuh>
#endif

namespace at {

CUDALongTensor::CUDALongTensor(Context* context)
: CUDALongTensor(context,THCudaLongTensor_new(context->thc_state)) {}

CUDALongTensor::CUDALongTensor(Context* context, THCudaLongTensor * tensor)
: TensorImpl(&context->getType(Backend::CUDA,ScalarType::Long)),
  tensor(tensor),
  context(context) {}
CUDALongTensor::~CUDALongTensor() {
  THCudaLongTensor_free(context->thc_state,  tensor);
}

const char * CUDALongTensor::toString() const {
  return "CUDALongTensor";
}

IntList CUDALongTensor::sizes() const {
  int64_t d = tensor->nDimension;
  if (d != 0) {
    // note: this will return "{}" for a scalar because dim() will return 0 in that case.
    return IntList(reinterpret_cast<int64_t*>(tensor->size),dim());
  } else {
    return IntList(kEmptySizes);
  }
}

int64_t CUDALongTensor::dim() const {
  if(isScalar())
    return 0;
  int64_t d = tensor->nDimension;
  // See Note [Empty versus 0-dim tensors]
  if (d != 0)
    return d;
  return kEmptySizes.size();
}

const char * CUDALongTensor::typeString() {
  return "CUDALongType";
}
void * CUDALongTensor::unsafeGetTH(bool retain) {
  if (retain)
      THCudaLongTensor_retain(context->thc_state,  tensor);
  return tensor;
}

// included as 'TensorDenseOrSparse' in TensorDerived.cpp

IntList CUDALongTensor::strides() const {
  int64_t d = tensor->nDimension;
  if (d != 0) {
    return IntList(reinterpret_cast<int64_t*>(tensor->stride),dim());
  } else {
    return IntList(kEmptyStrides);
  }
}
Scalar CUDALongTensor::localScalar() {
  int64_t numel = THCudaLongTensor_nElement(context->thc_state, tensor);
  AT_ASSERT(numel == 1,"localScalar() called on Tensor with %" PRId64 " elements",numel);
  return Scalar(int64_t(THCudaLongStorage_get(context->thc_state, tensor->storage, tensor->storageOffset)));
}
std::unique_ptr<Storage> CUDALongTensor::storage() {
  auto storage = THCudaLongTensor_storage(context->thc_state, tensor);
  THCudaLongStorage_retain(context->thc_state, storage);
  return std::unique_ptr<Storage>(new CUDALongStorage(&type().get_context(), storage));
}


}
