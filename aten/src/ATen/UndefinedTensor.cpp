#include "ATen/UndefinedTensor.h"
#include "ATen/Context.h"
#include "ATen/core/Error.h"

namespace at {

// should this use the globalContext?  Can it get a context passed in somehow?
UndefinedTensor::UndefinedTensor()
: TensorImpl(Backend::Undefined, ScalarType::Undefined, nullptr, /* is variable */ false) {
}

IntList UndefinedTensor::sizes() const {
  AT_ERROR("sizes() called on undefined Tensor");
}

int64_t UndefinedTensor::dim() const {
  AT_ERROR("dim() called on undefined Tensor");
}

void * UndefinedTensor::unsafeGetTH(bool retain) {
  AT_ERROR("unsafeGetTH(bool retain) called on undefined Tensor");
}
std::unique_ptr<Storage> UndefinedTensor::storage() {
  AT_ERROR("storage() called on undefined Tensor");
}

IntList UndefinedTensor::strides() const {
  AT_ERROR("strides() called on undefined Tensor");
}
UndefinedTensor UndefinedTensor::_singleton;

}
