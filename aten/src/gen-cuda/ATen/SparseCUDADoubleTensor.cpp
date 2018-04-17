// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/Config.h"
#include "ATen/SparseCUDADoubleTensor.h"
#include "ATen/CUDADoubleStorage.h"
#include "ATen/Scalar.h"
#include "ATen/Half.h"

#if AT_CUDA_ENABLED()
#include <ATen/cuda/CUDAHalf.cuh>
#endif

namespace at {

SparseCUDADoubleTensor::SparseCUDADoubleTensor(Context* context)
: SparseCUDADoubleTensor(context,THCSDoubleTensor_new(context->thc_state)) {}

SparseCUDADoubleTensor::SparseCUDADoubleTensor(Context* context, THCSDoubleTensor * tensor)
: TensorImpl(&context->getType(Backend::SparseCUDA,ScalarType::Double)),
  tensor(tensor),
  context(context) {}
SparseCUDADoubleTensor::~SparseCUDADoubleTensor() {
  THCSDoubleTensor_free(context->thc_state,  tensor);
}

const char * SparseCUDADoubleTensor::toString() const {
  return "SparseCUDADoubleTensor";
}

IntList SparseCUDADoubleTensor::sizes() const {
  int64_t d = tensor->nDimensionI + tensor->nDimensionV;
  if (d != 0) {
    // note: this will return "{}" for a scalar because dim() will return 0 in that case.
    return IntList(reinterpret_cast<int64_t*>(tensor->size),dim());
  } else {
    return IntList(kEmptySizes);
  }
}

int64_t SparseCUDADoubleTensor::dim() const {
  if(isScalar())
    return 0;
  int64_t d = tensor->nDimensionI + tensor->nDimensionV;
  // See Note [Empty versus 0-dim tensors]
  if (d != 0)
    return d;
  return kEmptySizes.size();
}

const char * SparseCUDADoubleTensor::typeString() {
  return "SparseCUDADoubleType";
}
void * SparseCUDADoubleTensor::unsafeGetTH(bool retain) {
  if (retain)
      THCSDoubleTensor_retain(context->thc_state,  tensor);
  return tensor;
}

// included as 'TensorDenseOrSparse' in TensorDerived.cpp
IntList SparseCUDADoubleTensor::strides() const {
 AT_ERROR("Sparse tensors do not have strides.");
}
Scalar SparseCUDADoubleTensor::localScalar() {
 AT_ERROR("NYI localScalar() on sparse tensors.");
}
std::unique_ptr<Storage> SparseCUDADoubleTensor::storage() {
  AT_ERROR("storage() is not implemented for %s", type().toString());
}


}
