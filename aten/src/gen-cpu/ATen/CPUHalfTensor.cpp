// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/Config.h"
#include "ATen/CPUHalfTensor.h"
#include "ATen/CPUHalfStorage.h"
#include "ATen/Scalar.h"
#include "ATen/Half.h"

#if AT_CUDA_ENABLED()

#endif

namespace at {

CPUHalfTensor::CPUHalfTensor(Context* context)
: CPUHalfTensor(context,THHalfTensor_new()) {}

CPUHalfTensor::CPUHalfTensor(Context* context, THHalfTensor * tensor)
: TensorImpl(&context->getType(Backend::CPU,ScalarType::Half)),
  tensor(tensor),
  context(context) {}
CPUHalfTensor::~CPUHalfTensor() {
  THHalfTensor_free( tensor);
}

const char * CPUHalfTensor::toString() const {
  return "CPUHalfTensor";
}

IntList CPUHalfTensor::sizes() const {
  int64_t d = tensor->nDimension;
  if (d != 0) {
    // note: this will return "{}" for a scalar because dim() will return 0 in that case.
    return IntList(reinterpret_cast<int64_t*>(tensor->size),dim());
  } else {
    return IntList(kEmptySizes);
  }
}

int64_t CPUHalfTensor::dim() const {
  if(isScalar())
    return 0;
  int64_t d = tensor->nDimension;
  // See Note [Empty versus 0-dim tensors]
  if (d != 0)
    return d;
  return kEmptySizes.size();
}

const char * CPUHalfTensor::typeString() {
  return "CPUHalfType";
}
void * CPUHalfTensor::unsafeGetTH(bool retain) {
  if (retain)
      THHalfTensor_retain( tensor);
  return tensor;
}

// included as 'TensorDenseOrSparse' in TensorDerived.cpp

IntList CPUHalfTensor::strides() const {
  int64_t d = tensor->nDimension;
  if (d != 0) {
    return IntList(reinterpret_cast<int64_t*>(tensor->stride),dim());
  } else {
    return IntList(kEmptyStrides);
  }
}
Scalar CPUHalfTensor::localScalar() {
  int64_t numel = THHalfTensor_nElement(tensor);
  AT_ASSERT(numel == 1,"localScalar() called on Tensor with %" PRId64 " elements",numel);
  return Scalar(HalfFix<Half,THHalf>(THHalfStorage_get(tensor->storage, tensor->storageOffset)));
}
std::unique_ptr<Storage> CPUHalfTensor::storage() {
  auto storage = THHalfTensor_storage(tensor);
  THHalfStorage_retain(storage);
  return std::unique_ptr<Storage>(new CPUHalfStorage(&type().get_context(), storage));
}


}
