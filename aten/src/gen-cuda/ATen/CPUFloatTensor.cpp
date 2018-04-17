// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/Config.h"
#include "ATen/CPUFloatTensor.h"
#include "ATen/CPUFloatStorage.h"
#include "ATen/Scalar.h"
#include "ATen/Half.h"

#if AT_CUDA_ENABLED()

#endif

namespace at {

CPUFloatTensor::CPUFloatTensor(Context* context)
: CPUFloatTensor(context,THFloatTensor_new()) {}

CPUFloatTensor::CPUFloatTensor(Context* context, THFloatTensor * tensor)
: TensorImpl(&context->getType(Backend::CPU,ScalarType::Float)),
  tensor(tensor),
  context(context) {}
CPUFloatTensor::~CPUFloatTensor() {
  THFloatTensor_free( tensor);
}

const char * CPUFloatTensor::toString() const {
  return "CPUFloatTensor";
}

IntList CPUFloatTensor::sizes() const {
  int64_t d = tensor->nDimension;
  if (d != 0) {
    // note: this will return "{}" for a scalar because dim() will return 0 in that case.
    return IntList(reinterpret_cast<int64_t*>(tensor->size),dim());
  } else {
    return IntList(kEmptySizes);
  }
}

int64_t CPUFloatTensor::dim() const {
  if(isScalar())
    return 0;
  int64_t d = tensor->nDimension;
  // See Note [Empty versus 0-dim tensors]
  if (d != 0)
    return d;
  return kEmptySizes.size();
}

const char * CPUFloatTensor::typeString() {
  return "CPUFloatType";
}
void * CPUFloatTensor::unsafeGetTH(bool retain) {
  if (retain)
      THFloatTensor_retain( tensor);
  return tensor;
}

// included as 'TensorDenseOrSparse' in TensorDerived.cpp

IntList CPUFloatTensor::strides() const {
  int64_t d = tensor->nDimension;
  if (d != 0) {
    return IntList(reinterpret_cast<int64_t*>(tensor->stride),dim());
  } else {
    return IntList(kEmptyStrides);
  }
}
Scalar CPUFloatTensor::localScalar() {
  int64_t numel = THFloatTensor_nElement(tensor);
  AT_ASSERT(numel == 1,"localScalar() called on Tensor with %" PRId64 " elements",numel);
  return Scalar((THFloatStorage_get(tensor->storage, tensor->storageOffset)));
}
std::unique_ptr<Storage> CPUFloatTensor::storage() {
  auto storage = THFloatTensor_storage(tensor);
  THFloatStorage_retain(storage);
  return std::unique_ptr<Storage>(new CPUFloatStorage(&type().get_context(), storage));
}


}
