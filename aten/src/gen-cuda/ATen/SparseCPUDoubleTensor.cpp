// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/Config.h"
#include "ATen/SparseCPUDoubleTensor.h"
#include "ATen/CPUDoubleStorage.h"
#include "ATen/Scalar.h"
#include "ATen/Half.h"

#if AT_CUDA_ENABLED()

#endif

namespace at {

SparseCPUDoubleTensor::SparseCPUDoubleTensor(Context* context)
: SparseCPUDoubleTensor(context,THSDoubleTensor_new()) {}

SparseCPUDoubleTensor::SparseCPUDoubleTensor(Context* context, THSDoubleTensor * tensor)
: TensorImpl(&context->getType(Backend::SparseCPU,ScalarType::Double)),
  tensor(tensor),
  context(context) {}
SparseCPUDoubleTensor::~SparseCPUDoubleTensor() {
  THSDoubleTensor_free( tensor);
}

const char * SparseCPUDoubleTensor::toString() const {
  return "SparseCPUDoubleTensor";
}

IntList SparseCPUDoubleTensor::sizes() const {
  int64_t d = tensor->nDimensionI + tensor->nDimensionV;
  if (d != 0) {
    // note: this will return "{}" for a scalar because dim() will return 0 in that case.
    return IntList(reinterpret_cast<int64_t*>(tensor->size),dim());
  } else {
    return IntList(kEmptySizes);
  }
}

int64_t SparseCPUDoubleTensor::dim() const {
  if(isScalar())
    return 0;
  int64_t d = tensor->nDimensionI + tensor->nDimensionV;
  // See Note [Empty versus 0-dim tensors]
  if (d != 0)
    return d;
  return kEmptySizes.size();
}

const char * SparseCPUDoubleTensor::typeString() {
  return "SparseCPUDoubleType";
}
void * SparseCPUDoubleTensor::unsafeGetTH(bool retain) {
  if (retain)
      THSDoubleTensor_retain( tensor);
  return tensor;
}

// included as 'TensorDenseOrSparse' in TensorDerived.cpp
IntList SparseCPUDoubleTensor::strides() const {
 AT_ERROR("Sparse tensors do not have strides.");
}
Scalar SparseCPUDoubleTensor::localScalar() {
 AT_ERROR("NYI localScalar() on sparse tensors.");
}
std::unique_ptr<Storage> SparseCPUDoubleTensor::storage() {
  AT_ERROR("storage() is not implemented for %s", type().toString());
}


}
