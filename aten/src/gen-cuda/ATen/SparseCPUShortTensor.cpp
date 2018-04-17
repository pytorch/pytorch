// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/Config.h"
#include "ATen/SparseCPUShortTensor.h"
#include "ATen/CPUShortStorage.h"
#include "ATen/Scalar.h"
#include "ATen/Half.h"

#if AT_CUDA_ENABLED()

#endif

namespace at {

SparseCPUShortTensor::SparseCPUShortTensor(Context* context)
: SparseCPUShortTensor(context,THSShortTensor_new()) {}

SparseCPUShortTensor::SparseCPUShortTensor(Context* context, THSShortTensor * tensor)
: TensorImpl(&context->getType(Backend::SparseCPU,ScalarType::Short)),
  tensor(tensor),
  context(context) {}
SparseCPUShortTensor::~SparseCPUShortTensor() {
  THSShortTensor_free( tensor);
}

const char * SparseCPUShortTensor::toString() const {
  return "SparseCPUShortTensor";
}

IntList SparseCPUShortTensor::sizes() const {
  int64_t d = tensor->nDimensionI + tensor->nDimensionV;
  if (d != 0) {
    // note: this will return "{}" for a scalar because dim() will return 0 in that case.
    return IntList(reinterpret_cast<int64_t*>(tensor->size),dim());
  } else {
    return IntList(kEmptySizes);
  }
}

int64_t SparseCPUShortTensor::dim() const {
  if(isScalar())
    return 0;
  int64_t d = tensor->nDimensionI + tensor->nDimensionV;
  // See Note [Empty versus 0-dim tensors]
  if (d != 0)
    return d;
  return kEmptySizes.size();
}

const char * SparseCPUShortTensor::typeString() {
  return "SparseCPUShortType";
}
void * SparseCPUShortTensor::unsafeGetTH(bool retain) {
  if (retain)
      THSShortTensor_retain( tensor);
  return tensor;
}

// included as 'TensorDenseOrSparse' in TensorDerived.cpp
IntList SparseCPUShortTensor::strides() const {
 AT_ERROR("Sparse tensors do not have strides.");
}
Scalar SparseCPUShortTensor::localScalar() {
 AT_ERROR("NYI localScalar() on sparse tensors.");
}
std::unique_ptr<Storage> SparseCPUShortTensor::storage() {
  AT_ERROR("storage() is not implemented for %s", type().toString());
}


}
