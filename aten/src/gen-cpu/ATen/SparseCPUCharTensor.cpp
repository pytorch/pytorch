// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/Config.h"
#include "ATen/SparseCPUCharTensor.h"
#include "ATen/CPUCharStorage.h"
#include "ATen/Scalar.h"
#include "ATen/Half.h"

#if AT_CUDA_ENABLED()

#endif

namespace at {

SparseCPUCharTensor::SparseCPUCharTensor(Context* context)
: SparseCPUCharTensor(context,THSCharTensor_new()) {}

SparseCPUCharTensor::SparseCPUCharTensor(Context* context, THSCharTensor * tensor)
: TensorImpl(&context->getType(Backend::SparseCPU,ScalarType::Char)),
  tensor(tensor),
  context(context) {}
SparseCPUCharTensor::~SparseCPUCharTensor() {
  THSCharTensor_free( tensor);
}

const char * SparseCPUCharTensor::toString() const {
  return "SparseCPUCharTensor";
}

IntList SparseCPUCharTensor::sizes() const {
  int64_t d = tensor->nDimensionI + tensor->nDimensionV;
  if (d != 0) {
    // note: this will return "{}" for a scalar because dim() will return 0 in that case.
    return IntList(reinterpret_cast<int64_t*>(tensor->size),dim());
  } else {
    return IntList(kEmptySizes);
  }
}

int64_t SparseCPUCharTensor::dim() const {
  if(isScalar())
    return 0;
  int64_t d = tensor->nDimensionI + tensor->nDimensionV;
  // See Note [Empty versus 0-dim tensors]
  if (d != 0)
    return d;
  return kEmptySizes.size();
}

const char * SparseCPUCharTensor::typeString() {
  return "SparseCPUCharType";
}
void * SparseCPUCharTensor::unsafeGetTH(bool retain) {
  if (retain)
      THSCharTensor_retain( tensor);
  return tensor;
}

// included as 'TensorDenseOrSparse' in TensorDerived.cpp
IntList SparseCPUCharTensor::strides() const {
 AT_ERROR("Sparse tensors do not have strides.");
}
Scalar SparseCPUCharTensor::localScalar() {
 AT_ERROR("NYI localScalar() on sparse tensors.");
}
std::unique_ptr<Storage> SparseCPUCharTensor::storage() {
  AT_ERROR("storage() is not implemented for %s", type().toString());
}


}
