// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/Config.h"
#include "ATen/CPUShortTensor.h"
#include "ATen/CPUShortStorage.h"
#include "ATen/Scalar.h"
#include "ATen/Half.h"

#if AT_CUDA_ENABLED()

#endif

namespace at {

CPUShortTensor::CPUShortTensor(Context* context)
: CPUShortTensor(context,THShortTensor_new()) {}

CPUShortTensor::CPUShortTensor(Context* context, THShortTensor * tensor)
: TensorImpl(&context->getType(Backend::CPU,ScalarType::Short)),
  tensor(tensor),
  context(context) {}
CPUShortTensor::~CPUShortTensor() {
  THShortTensor_free( tensor);
}

const char * CPUShortTensor::toString() const {
  return "CPUShortTensor";
}

IntList CPUShortTensor::sizes() const {
  int64_t d = tensor->nDimension;
  if (d != 0) {
    // note: this will return "{}" for a scalar because dim() will return 0 in that case.
    return IntList(reinterpret_cast<int64_t*>(tensor->size),dim());
  } else {
    return IntList(kEmptySizes);
  }
}

int64_t CPUShortTensor::dim() const {
  if(isScalar())
    return 0;
  int64_t d = tensor->nDimension;
  // See Note [Empty versus 0-dim tensors]
  if (d != 0)
    return d;
  return kEmptySizes.size();
}

const char * CPUShortTensor::typeString() {
  return "CPUShortType";
}
void * CPUShortTensor::unsafeGetTH(bool retain) {
  if (retain)
      THShortTensor_retain( tensor);
  return tensor;
}

// included as 'TensorDenseOrSparse' in TensorDerived.cpp

IntList CPUShortTensor::strides() const {
  int64_t d = tensor->nDimension;
  if (d != 0) {
    return IntList(reinterpret_cast<int64_t*>(tensor->stride),dim());
  } else {
    return IntList(kEmptyStrides);
  }
}
Scalar CPUShortTensor::localScalar() {
  int64_t numel = THShortTensor_nElement(tensor);
  AT_ASSERT(numel == 1,"localScalar() called on Tensor with %" PRId64 " elements",numel);
  return Scalar((THShortStorage_get(tensor->storage, tensor->storageOffset)));
}
std::unique_ptr<Storage> CPUShortTensor::storage() {
  auto storage = THShortTensor_storage(tensor);
  THShortStorage_retain(storage);
  return std::unique_ptr<Storage>(new CPUShortStorage(&type().get_context(), storage));
}


}
