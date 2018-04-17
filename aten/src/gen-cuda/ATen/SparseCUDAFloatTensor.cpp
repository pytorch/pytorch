// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/Config.h"
#include "ATen/SparseCUDAFloatTensor.h"
#include "ATen/CUDAFloatStorage.h"
#include "ATen/Scalar.h"
#include "ATen/Half.h"

#if AT_CUDA_ENABLED()
#include <ATen/cuda/CUDAHalf.cuh>
#endif

namespace at {

SparseCUDAFloatTensor::SparseCUDAFloatTensor(Context* context)
: SparseCUDAFloatTensor(context,THCSFloatTensor_new(context->thc_state)) {}

SparseCUDAFloatTensor::SparseCUDAFloatTensor(Context* context, THCSFloatTensor * tensor)
: TensorImpl(&context->getType(Backend::SparseCUDA,ScalarType::Float)),
  tensor(tensor),
  context(context) {}
SparseCUDAFloatTensor::~SparseCUDAFloatTensor() {
  THCSFloatTensor_free(context->thc_state,  tensor);
}

const char * SparseCUDAFloatTensor::toString() const {
  return "SparseCUDAFloatTensor";
}

IntList SparseCUDAFloatTensor::sizes() const {
  int64_t d = tensor->nDimensionI + tensor->nDimensionV;
  if (d != 0) {
    // note: this will return "{}" for a scalar because dim() will return 0 in that case.
    return IntList(reinterpret_cast<int64_t*>(tensor->size),dim());
  } else {
    return IntList(kEmptySizes);
  }
}

int64_t SparseCUDAFloatTensor::dim() const {
  if(isScalar())
    return 0;
  int64_t d = tensor->nDimensionI + tensor->nDimensionV;
  // See Note [Empty versus 0-dim tensors]
  if (d != 0)
    return d;
  return kEmptySizes.size();
}

const char * SparseCUDAFloatTensor::typeString() {
  return "SparseCUDAFloatType";
}
void * SparseCUDAFloatTensor::unsafeGetTH(bool retain) {
  if (retain)
      THCSFloatTensor_retain(context->thc_state,  tensor);
  return tensor;
}

// included as 'TensorDenseOrSparse' in TensorDerived.cpp
IntList SparseCUDAFloatTensor::strides() const {
 AT_ERROR("Sparse tensors do not have strides.");
}
Scalar SparseCUDAFloatTensor::localScalar() {
 AT_ERROR("NYI localScalar() on sparse tensors.");
}
std::unique_ptr<Storage> SparseCUDAFloatTensor::storage() {
  AT_ERROR("storage() is not implemented for %s", type().toString());
}


}
