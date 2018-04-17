// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/Config.h"
#include "ATen/SparseCPUIntTensor.h"
#include "ATen/CPUIntStorage.h"
#include "ATen/Scalar.h"
#include "ATen/Half.h"

#if AT_CUDA_ENABLED()

#endif

namespace at {

SparseCPUIntTensor::SparseCPUIntTensor(Context* context)
: SparseCPUIntTensor(context,THSIntTensor_new()) {}

SparseCPUIntTensor::SparseCPUIntTensor(Context* context, THSIntTensor * tensor)
: TensorImpl(&context->getType(Backend::SparseCPU,ScalarType::Int)),
  tensor(tensor),
  context(context) {}
SparseCPUIntTensor::~SparseCPUIntTensor() {
  THSIntTensor_free( tensor);
}

const char * SparseCPUIntTensor::toString() const {
  return "SparseCPUIntTensor";
}

IntList SparseCPUIntTensor::sizes() const {
  int64_t d = tensor->nDimensionI + tensor->nDimensionV;
  if (d != 0) {
    // note: this will return "{}" for a scalar because dim() will return 0 in that case.
    return IntList(reinterpret_cast<int64_t*>(tensor->size),dim());
  } else {
    return IntList(kEmptySizes);
  }
}

int64_t SparseCPUIntTensor::dim() const {
  if(isScalar())
    return 0;
  int64_t d = tensor->nDimensionI + tensor->nDimensionV;
  // See Note [Empty versus 0-dim tensors]
  if (d != 0)
    return d;
  return kEmptySizes.size();
}

const char * SparseCPUIntTensor::typeString() {
  return "SparseCPUIntType";
}
void * SparseCPUIntTensor::unsafeGetTH(bool retain) {
  if (retain)
      THSIntTensor_retain( tensor);
  return tensor;
}

// included as 'TensorDenseOrSparse' in TensorDerived.cpp
IntList SparseCPUIntTensor::strides() const {
 AT_ERROR("Sparse tensors do not have strides.");
}
Scalar SparseCPUIntTensor::localScalar() {
 AT_ERROR("NYI localScalar() on sparse tensors.");
}
std::unique_ptr<Storage> SparseCPUIntTensor::storage() {
  AT_ERROR("storage() is not implemented for %s", type().toString());
}


}
