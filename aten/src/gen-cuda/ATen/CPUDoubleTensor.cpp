// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/Config.h"
#include "ATen/CPUDoubleTensor.h"
#include "ATen/CPUDoubleStorage.h"
#include "ATen/Scalar.h"
#include "ATen/Half.h"

#if AT_CUDA_ENABLED()

#endif

namespace at {

CPUDoubleTensor::CPUDoubleTensor(Context* context)
: CPUDoubleTensor(context,THDoubleTensor_new()) {}

CPUDoubleTensor::CPUDoubleTensor(Context* context, THDoubleTensor * tensor)
: TensorImpl(&context->getType(Backend::CPU,ScalarType::Double)),
  tensor(tensor),
  context(context) {}
CPUDoubleTensor::~CPUDoubleTensor() {
  THDoubleTensor_free( tensor);
}

const char * CPUDoubleTensor::toString() const {
  return "CPUDoubleTensor";
}

IntList CPUDoubleTensor::sizes() const {
  int64_t d = tensor->nDimension;
  if (d != 0) {
    // note: this will return "{}" for a scalar because dim() will return 0 in that case.
    return IntList(reinterpret_cast<int64_t*>(tensor->size),dim());
  } else {
    return IntList(kEmptySizes);
  }
}

int64_t CPUDoubleTensor::dim() const {
  if(isScalar())
    return 0;
  int64_t d = tensor->nDimension;
  // See Note [Empty versus 0-dim tensors]
  if (d != 0)
    return d;
  return kEmptySizes.size();
}

const char * CPUDoubleTensor::typeString() {
  return "CPUDoubleType";
}
void * CPUDoubleTensor::unsafeGetTH(bool retain) {
  if (retain)
      THDoubleTensor_retain( tensor);
  return tensor;
}

// included as 'TensorDenseOrSparse' in TensorDerived.cpp

IntList CPUDoubleTensor::strides() const {
  int64_t d = tensor->nDimension;
  if (d != 0) {
    return IntList(reinterpret_cast<int64_t*>(tensor->stride),dim());
  } else {
    return IntList(kEmptyStrides);
  }
}
Scalar CPUDoubleTensor::localScalar() {
  int64_t numel = THDoubleTensor_nElement(tensor);
  AT_ASSERT(numel == 1,"localScalar() called on Tensor with %" PRId64 " elements",numel);
  return Scalar((THDoubleStorage_get(tensor->storage, tensor->storageOffset)));
}
std::unique_ptr<Storage> CPUDoubleTensor::storage() {
  auto storage = THDoubleTensor_storage(tensor);
  THDoubleStorage_retain(storage);
  return std::unique_ptr<Storage>(new CPUDoubleStorage(&type().get_context(), storage));
}


}
