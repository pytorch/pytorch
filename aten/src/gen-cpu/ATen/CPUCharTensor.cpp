// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/Config.h"
#include "ATen/CPUCharTensor.h"
#include "ATen/CPUCharStorage.h"
#include "ATen/Scalar.h"
#include "ATen/Half.h"

#if AT_CUDA_ENABLED()

#endif

namespace at {

CPUCharTensor::CPUCharTensor(Context* context)
: CPUCharTensor(context,THCharTensor_new()) {}

CPUCharTensor::CPUCharTensor(Context* context, THCharTensor * tensor)
: TensorImpl(&context->getType(Backend::CPU,ScalarType::Char)),
  tensor(tensor),
  context(context) {}
CPUCharTensor::~CPUCharTensor() {
  THCharTensor_free( tensor);
}

const char * CPUCharTensor::toString() const {
  return "CPUCharTensor";
}

IntList CPUCharTensor::sizes() const {
  int64_t d = tensor->nDimension;
  if (d != 0) {
    // note: this will return "{}" for a scalar because dim() will return 0 in that case.
    return IntList(reinterpret_cast<int64_t*>(tensor->size),dim());
  } else {
    return IntList(kEmptySizes);
  }
}

int64_t CPUCharTensor::dim() const {
  if(isScalar())
    return 0;
  int64_t d = tensor->nDimension;
  // See Note [Empty versus 0-dim tensors]
  if (d != 0)
    return d;
  return kEmptySizes.size();
}

const char * CPUCharTensor::typeString() {
  return "CPUCharType";
}
void * CPUCharTensor::unsafeGetTH(bool retain) {
  if (retain)
      THCharTensor_retain( tensor);
  return tensor;
}

// included as 'TensorDenseOrSparse' in TensorDerived.cpp

IntList CPUCharTensor::strides() const {
  int64_t d = tensor->nDimension;
  if (d != 0) {
    return IntList(reinterpret_cast<int64_t*>(tensor->stride),dim());
  } else {
    return IntList(kEmptyStrides);
  }
}
Scalar CPUCharTensor::localScalar() {
  int64_t numel = THCharTensor_nElement(tensor);
  AT_ASSERT(numel == 1,"localScalar() called on Tensor with %" PRId64 " elements",numel);
  return Scalar((THCharStorage_get(tensor->storage, tensor->storageOffset)));
}
std::unique_ptr<Storage> CPUCharTensor::storage() {
  auto storage = THCharTensor_storage(tensor);
  THCharStorage_retain(storage);
  return std::unique_ptr<Storage>(new CPUCharStorage(&type().get_context(), storage));
}


}
