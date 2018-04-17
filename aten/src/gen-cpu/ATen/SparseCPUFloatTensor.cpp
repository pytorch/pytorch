// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/Config.h"
#include "ATen/SparseCPUFloatTensor.h"
#include "ATen/CPUFloatStorage.h"
#include "ATen/Scalar.h"
#include "ATen/Half.h"

#if AT_CUDA_ENABLED()

#endif

namespace at {

SparseCPUFloatTensor::SparseCPUFloatTensor(Context* context)
: SparseCPUFloatTensor(context,THSFloatTensor_new()) {}

SparseCPUFloatTensor::SparseCPUFloatTensor(Context* context, THSFloatTensor * tensor)
: TensorImpl(&context->getType(Backend::SparseCPU,ScalarType::Float)),
  tensor(tensor),
  context(context) {}
SparseCPUFloatTensor::~SparseCPUFloatTensor() {
  THSFloatTensor_free( tensor);
}

const char * SparseCPUFloatTensor::toString() const {
  return "SparseCPUFloatTensor";
}

IntList SparseCPUFloatTensor::sizes() const {
  int64_t d = tensor->nDimensionI + tensor->nDimensionV;
  if (d != 0) {
    // note: this will return "{}" for a scalar because dim() will return 0 in that case.
    return IntList(reinterpret_cast<int64_t*>(tensor->size),dim());
  } else {
    return IntList(kEmptySizes);
  }
}

int64_t SparseCPUFloatTensor::dim() const {
  if(isScalar())
    return 0;
  int64_t d = tensor->nDimensionI + tensor->nDimensionV;
  // See Note [Empty versus 0-dim tensors]
  if (d != 0)
    return d;
  return kEmptySizes.size();
}

const char * SparseCPUFloatTensor::typeString() {
  return "SparseCPUFloatType";
}
void * SparseCPUFloatTensor::unsafeGetTH(bool retain) {
  if (retain)
      THSFloatTensor_retain( tensor);
  return tensor;
}

// included as 'TensorDenseOrSparse' in TensorDerived.cpp
IntList SparseCPUFloatTensor::strides() const {
 AT_ERROR("Sparse tensors do not have strides.");
}
Scalar SparseCPUFloatTensor::localScalar() {
 AT_ERROR("NYI localScalar() on sparse tensors.");
}
std::unique_ptr<Storage> SparseCPUFloatTensor::storage() {
  AT_ERROR("storage() is not implemented for %s", type().toString());
}


}
