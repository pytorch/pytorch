#include "ATen/UndefinedTensor.h"
#include "ATen/Context.h"
#include "ATen/core/Error.h"

namespace at {

// should this use the globalContext?  Can it get a context passed in somehow?
UndefinedTensor::UndefinedTensor()
: TensorImpl(UndefinedTensorId(), ScalarType::Undefined,  /* is variable */ false) {
}

IntList UndefinedTensor::sizes() const {
  AT_ERROR("sizes() called on undefined Tensor");
}

int64_t UndefinedTensor::size(int64_t d) const {
  AT_ERROR("size(dim) called on an undefined Tensor");
}

int64_t UndefinedTensor::stride(int64_t d) const {
  AT_ERROR("stride(dim) called on an undefined Tensor");
}

int64_t UndefinedTensor::dim() const {
  AT_ERROR("dim() called on undefined Tensor");
}

std::unique_ptr<Storage> UndefinedTensor::storage() {
  AT_ERROR("storage() called on undefined Tensor");
}

at::StorageImpl* UndefinedTensor::storageImpl() const {
  AT_ERROR("storageImpl() called on an undefined Tensor");
}

int64_t UndefinedTensor::storage_offset() const {
  AT_ERROR("storage_offset() called on an undefined Tensor");
}

IntList UndefinedTensor::strides() const {
  AT_ERROR("strides() called on undefined Tensor");
}
UndefinedTensor UndefinedTensor::_singleton;

}
