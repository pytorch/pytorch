// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/Config.h"
#include "ATen/SparseCPUByteTensor.h"
#include "ATen/CPUByteStorage.h"
#include "ATen/Scalar.h"
#include "ATen/Half.h"

#if AT_CUDA_ENABLED()

#endif

namespace at {

SparseCPUByteTensor::SparseCPUByteTensor(Context* context)
: SparseCPUByteTensor(context,THSByteTensor_new()) {}

SparseCPUByteTensor::SparseCPUByteTensor(Context* context, THSByteTensor * tensor)
: TensorImpl(&context->getType(Backend::SparseCPU,ScalarType::Byte)),
  tensor(tensor),
  context(context) {}
SparseCPUByteTensor::~SparseCPUByteTensor() {
  THSByteTensor_free( tensor);
}

const char * SparseCPUByteTensor::toString() const {
  return "SparseCPUByteTensor";
}

IntList SparseCPUByteTensor::sizes() const {
  int64_t d = tensor->nDimensionI + tensor->nDimensionV;
  if (d != 0) {
    // note: this will return "{}" for a scalar because dim() will return 0 in that case.
    return IntList(reinterpret_cast<int64_t*>(tensor->size),dim());
  } else {
    return IntList(kEmptySizes);
  }
}

int64_t SparseCPUByteTensor::dim() const {
  if(isScalar())
    return 0;
  int64_t d = tensor->nDimensionI + tensor->nDimensionV;
  // See Note [Empty versus 0-dim tensors]
  if (d != 0)
    return d;
  return kEmptySizes.size();
}

const char * SparseCPUByteTensor::typeString() {
  return "SparseCPUByteType";
}
void * SparseCPUByteTensor::unsafeGetTH(bool retain) {
  if (retain)
      THSByteTensor_retain( tensor);
  return tensor;
}

// included as 'TensorDenseOrSparse' in TensorDerived.cpp
IntList SparseCPUByteTensor::strides() const {
 AT_ERROR("Sparse tensors do not have strides.");
}
Scalar SparseCPUByteTensor::localScalar() {
 AT_ERROR("NYI localScalar() on sparse tensors.");
}
std::unique_ptr<Storage> SparseCPUByteTensor::storage() {
  AT_ERROR("storage() is not implemented for %s", type().toString());
}


}
