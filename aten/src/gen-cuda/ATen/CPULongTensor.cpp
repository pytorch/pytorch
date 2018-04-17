// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/Config.h"
#include "ATen/CPULongTensor.h"
#include "ATen/CPULongStorage.h"
#include "ATen/Scalar.h"
#include "ATen/Half.h"

#if AT_CUDA_ENABLED()

#endif

namespace at {

CPULongTensor::CPULongTensor(Context* context)
: CPULongTensor(context,THLongTensor_new()) {}

CPULongTensor::CPULongTensor(Context* context, THLongTensor * tensor)
: TensorImpl(&context->getType(Backend::CPU,ScalarType::Long)),
  tensor(tensor),
  context(context) {}
CPULongTensor::~CPULongTensor() {
  THLongTensor_free( tensor);
}

const char * CPULongTensor::toString() const {
  return "CPULongTensor";
}

IntList CPULongTensor::sizes() const {
  int64_t d = tensor->nDimension;
  if (d != 0) {
    // note: this will return "{}" for a scalar because dim() will return 0 in that case.
    return IntList(reinterpret_cast<int64_t*>(tensor->size),dim());
  } else {
    return IntList(kEmptySizes);
  }
}

int64_t CPULongTensor::dim() const {
  if(isScalar())
    return 0;
  int64_t d = tensor->nDimension;
  // See Note [Empty versus 0-dim tensors]
  if (d != 0)
    return d;
  return kEmptySizes.size();
}

const char * CPULongTensor::typeString() {
  return "CPULongType";
}
void * CPULongTensor::unsafeGetTH(bool retain) {
  if (retain)
      THLongTensor_retain( tensor);
  return tensor;
}

// included as 'TensorDenseOrSparse' in TensorDerived.cpp

IntList CPULongTensor::strides() const {
  int64_t d = tensor->nDimension;
  if (d != 0) {
    return IntList(reinterpret_cast<int64_t*>(tensor->stride),dim());
  } else {
    return IntList(kEmptyStrides);
  }
}
Scalar CPULongTensor::localScalar() {
  int64_t numel = THLongTensor_nElement(tensor);
  AT_ASSERT(numel == 1,"localScalar() called on Tensor with %" PRId64 " elements",numel);
  return Scalar(int64_t(THLongStorage_get(tensor->storage, tensor->storageOffset)));
}
std::unique_ptr<Storage> CPULongTensor::storage() {
  auto storage = THLongTensor_storage(tensor);
  THLongStorage_retain(storage);
  return std::unique_ptr<Storage>(new CPULongStorage(&type().get_context(), storage));
}


}
