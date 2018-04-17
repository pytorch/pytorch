// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/Config.h"
#include "ATen/SparseCUDALongTensor.h"
#include "ATen/CUDALongStorage.h"
#include "ATen/Scalar.h"
#include "ATen/Half.h"

#if AT_CUDA_ENABLED()
#include <ATen/cuda/CUDAHalf.cuh>
#endif

namespace at {

SparseCUDALongTensor::SparseCUDALongTensor(Context* context)
: SparseCUDALongTensor(context,THCSLongTensor_new(context->thc_state)) {}

SparseCUDALongTensor::SparseCUDALongTensor(Context* context, THCSLongTensor * tensor)
: TensorImpl(&context->getType(Backend::SparseCUDA,ScalarType::Long)),
  tensor(tensor),
  context(context) {}
SparseCUDALongTensor::~SparseCUDALongTensor() {
  THCSLongTensor_free(context->thc_state,  tensor);
}

const char * SparseCUDALongTensor::toString() const {
  return "SparseCUDALongTensor";
}

IntList SparseCUDALongTensor::sizes() const {
  int64_t d = tensor->nDimensionI + tensor->nDimensionV;
  if (d != 0) {
    // note: this will return "{}" for a scalar because dim() will return 0 in that case.
    return IntList(reinterpret_cast<int64_t*>(tensor->size),dim());
  } else {
    return IntList(kEmptySizes);
  }
}

int64_t SparseCUDALongTensor::dim() const {
  if(isScalar())
    return 0;
  int64_t d = tensor->nDimensionI + tensor->nDimensionV;
  // See Note [Empty versus 0-dim tensors]
  if (d != 0)
    return d;
  return kEmptySizes.size();
}

const char * SparseCUDALongTensor::typeString() {
  return "SparseCUDALongType";
}
void * SparseCUDALongTensor::unsafeGetTH(bool retain) {
  if (retain)
      THCSLongTensor_retain(context->thc_state,  tensor);
  return tensor;
}

// included as 'TensorDenseOrSparse' in TensorDerived.cpp
IntList SparseCUDALongTensor::strides() const {
 AT_ERROR("Sparse tensors do not have strides.");
}
Scalar SparseCUDALongTensor::localScalar() {
 AT_ERROR("NYI localScalar() on sparse tensors.");
}
std::unique_ptr<Storage> SparseCUDALongTensor::storage() {
  AT_ERROR("storage() is not implemented for %s", type().toString());
}


}
