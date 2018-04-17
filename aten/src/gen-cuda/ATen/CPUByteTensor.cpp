// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/Config.h"
#include "ATen/CPUByteTensor.h"
#include "ATen/CPUByteStorage.h"
#include "ATen/Scalar.h"
#include "ATen/Half.h"

#if AT_CUDA_ENABLED()

#endif

namespace at {

CPUByteTensor::CPUByteTensor(Context* context)
: CPUByteTensor(context,THByteTensor_new()) {}

CPUByteTensor::CPUByteTensor(Context* context, THByteTensor * tensor)
: TensorImpl(&context->getType(Backend::CPU,ScalarType::Byte)),
  tensor(tensor),
  context(context) {}
CPUByteTensor::~CPUByteTensor() {
  THByteTensor_free( tensor);
}

const char * CPUByteTensor::toString() const {
  return "CPUByteTensor";
}

IntList CPUByteTensor::sizes() const {
  int64_t d = tensor->nDimension;
  if (d != 0) {
    // note: this will return "{}" for a scalar because dim() will return 0 in that case.
    return IntList(reinterpret_cast<int64_t*>(tensor->size),dim());
  } else {
    return IntList(kEmptySizes);
  }
}

int64_t CPUByteTensor::dim() const {
  if(isScalar())
    return 0;
  int64_t d = tensor->nDimension;
  // See Note [Empty versus 0-dim tensors]
  if (d != 0)
    return d;
  return kEmptySizes.size();
}

const char * CPUByteTensor::typeString() {
  return "CPUByteType";
}
void * CPUByteTensor::unsafeGetTH(bool retain) {
  if (retain)
      THByteTensor_retain( tensor);
  return tensor;
}

// included as 'TensorDenseOrSparse' in TensorDerived.cpp

IntList CPUByteTensor::strides() const {
  int64_t d = tensor->nDimension;
  if (d != 0) {
    return IntList(reinterpret_cast<int64_t*>(tensor->stride),dim());
  } else {
    return IntList(kEmptyStrides);
  }
}
Scalar CPUByteTensor::localScalar() {
  int64_t numel = THByteTensor_nElement(tensor);
  AT_ASSERT(numel == 1,"localScalar() called on Tensor with %" PRId64 " elements",numel);
  return Scalar((THByteStorage_get(tensor->storage, tensor->storageOffset)));
}
std::unique_ptr<Storage> CPUByteTensor::storage() {
  auto storage = THByteTensor_storage(tensor);
  THByteStorage_retain(storage);
  return std::unique_ptr<Storage>(new CPUByteStorage(&type().get_context(), storage));
}


}
