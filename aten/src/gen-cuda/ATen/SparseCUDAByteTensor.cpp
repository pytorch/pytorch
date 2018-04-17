// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/Config.h"
#include "ATen/SparseCUDAByteTensor.h"
#include "ATen/CUDAByteStorage.h"
#include "ATen/Scalar.h"
#include "ATen/Half.h"

#if AT_CUDA_ENABLED()
#include <ATen/cuda/CUDAHalf.cuh>
#endif

namespace at {

SparseCUDAByteTensor::SparseCUDAByteTensor(Context* context)
: SparseCUDAByteTensor(context,THCSByteTensor_new(context->thc_state)) {}

SparseCUDAByteTensor::SparseCUDAByteTensor(Context* context, THCSByteTensor * tensor)
: TensorImpl(&context->getType(Backend::SparseCUDA,ScalarType::Byte)),
  tensor(tensor),
  context(context) {}
SparseCUDAByteTensor::~SparseCUDAByteTensor() {
  THCSByteTensor_free(context->thc_state,  tensor);
}

const char * SparseCUDAByteTensor::toString() const {
  return "SparseCUDAByteTensor";
}

IntList SparseCUDAByteTensor::sizes() const {
  int64_t d = tensor->nDimensionI + tensor->nDimensionV;
  if (d != 0) {
    // note: this will return "{}" for a scalar because dim() will return 0 in that case.
    return IntList(reinterpret_cast<int64_t*>(tensor->size),dim());
  } else {
    return IntList(kEmptySizes);
  }
}

int64_t SparseCUDAByteTensor::dim() const {
  if(isScalar())
    return 0;
  int64_t d = tensor->nDimensionI + tensor->nDimensionV;
  // See Note [Empty versus 0-dim tensors]
  if (d != 0)
    return d;
  return kEmptySizes.size();
}

const char * SparseCUDAByteTensor::typeString() {
  return "SparseCUDAByteType";
}
void * SparseCUDAByteTensor::unsafeGetTH(bool retain) {
  if (retain)
      THCSByteTensor_retain(context->thc_state,  tensor);
  return tensor;
}

// included as 'TensorDenseOrSparse' in TensorDerived.cpp
IntList SparseCUDAByteTensor::strides() const {
 AT_ERROR("Sparse tensors do not have strides.");
}
Scalar SparseCUDAByteTensor::localScalar() {
 AT_ERROR("NYI localScalar() on sparse tensors.");
}
std::unique_ptr<Storage> SparseCUDAByteTensor::storage() {
  AT_ERROR("storage() is not implemented for %s", type().toString());
}


}
