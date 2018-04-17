// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/Config.h"
#include "ATen/SparseCPULongTensor.h"
#include "ATen/CPULongStorage.h"
#include "ATen/Scalar.h"
#include "ATen/Half.h"

#if AT_CUDA_ENABLED()

#endif

namespace at {

SparseCPULongTensor::SparseCPULongTensor(Context* context)
: SparseCPULongTensor(context,THSLongTensor_new()) {}

SparseCPULongTensor::SparseCPULongTensor(Context* context, THSLongTensor * tensor)
: TensorImpl(&context->getType(Backend::SparseCPU,ScalarType::Long)),
  tensor(tensor),
  context(context) {}
SparseCPULongTensor::~SparseCPULongTensor() {
  THSLongTensor_free( tensor);
}

const char * SparseCPULongTensor::toString() const {
  return "SparseCPULongTensor";
}

IntList SparseCPULongTensor::sizes() const {
  int64_t d = tensor->nDimensionI + tensor->nDimensionV;
  if (d != 0) {
    // note: this will return "{}" for a scalar because dim() will return 0 in that case.
    return IntList(reinterpret_cast<int64_t*>(tensor->size),dim());
  } else {
    return IntList(kEmptySizes);
  }
}

int64_t SparseCPULongTensor::dim() const {
  if(isScalar())
    return 0;
  int64_t d = tensor->nDimensionI + tensor->nDimensionV;
  // See Note [Empty versus 0-dim tensors]
  if (d != 0)
    return d;
  return kEmptySizes.size();
}

const char * SparseCPULongTensor::typeString() {
  return "SparseCPULongType";
}
void * SparseCPULongTensor::unsafeGetTH(bool retain) {
  if (retain)
      THSLongTensor_retain( tensor);
  return tensor;
}

// included as 'TensorDenseOrSparse' in TensorDerived.cpp
IntList SparseCPULongTensor::strides() const {
 AT_ERROR("Sparse tensors do not have strides.");
}
Scalar SparseCPULongTensor::localScalar() {
 AT_ERROR("NYI localScalar() on sparse tensors.");
}
std::unique_ptr<Storage> SparseCPULongTensor::storage() {
  AT_ERROR("storage() is not implemented for %s", type().toString());
}


}
