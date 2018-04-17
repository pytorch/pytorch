// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/Config.h"
#include "ATen/CPUIntTensor.h"
#include "ATen/CPUIntStorage.h"
#include "ATen/Scalar.h"
#include "ATen/Half.h"

#if AT_CUDA_ENABLED()

#endif

namespace at {

CPUIntTensor::CPUIntTensor(Context* context)
: CPUIntTensor(context,THIntTensor_new()) {}

CPUIntTensor::CPUIntTensor(Context* context, THIntTensor * tensor)
: TensorImpl(&context->getType(Backend::CPU,ScalarType::Int)),
  tensor(tensor),
  context(context) {}
CPUIntTensor::~CPUIntTensor() {
  THIntTensor_free( tensor);
}

const char * CPUIntTensor::toString() const {
  return "CPUIntTensor";
}

IntList CPUIntTensor::sizes() const {
  int64_t d = tensor->nDimension;
  if (d != 0) {
    // note: this will return "{}" for a scalar because dim() will return 0 in that case.
    return IntList(reinterpret_cast<int64_t*>(tensor->size),dim());
  } else {
    return IntList(kEmptySizes);
  }
}

int64_t CPUIntTensor::dim() const {
  if(isScalar())
    return 0;
  int64_t d = tensor->nDimension;
  // See Note [Empty versus 0-dim tensors]
  if (d != 0)
    return d;
  return kEmptySizes.size();
}

const char * CPUIntTensor::typeString() {
  return "CPUIntType";
}
void * CPUIntTensor::unsafeGetTH(bool retain) {
  if (retain)
      THIntTensor_retain( tensor);
  return tensor;
}

// included as 'TensorDenseOrSparse' in TensorDerived.cpp

IntList CPUIntTensor::strides() const {
  int64_t d = tensor->nDimension;
  if (d != 0) {
    return IntList(reinterpret_cast<int64_t*>(tensor->stride),dim());
  } else {
    return IntList(kEmptyStrides);
  }
}
Scalar CPUIntTensor::localScalar() {
  int64_t numel = THIntTensor_nElement(tensor);
  AT_ASSERT(numel == 1,"localScalar() called on Tensor with %" PRId64 " elements",numel);
  return Scalar((THIntStorage_get(tensor->storage, tensor->storageOffset)));
}
std::unique_ptr<Storage> CPUIntTensor::storage() {
  auto storage = THIntTensor_storage(tensor);
  THIntStorage_retain(storage);
  return std::unique_ptr<Storage>(new CPUIntStorage(&type().get_context(), storage));
}


}
