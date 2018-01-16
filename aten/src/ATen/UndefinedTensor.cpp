#include "ATen/UndefinedTensor.h"
#include "ATen/Context.h"

namespace at {

// should this use the globalContext?  Can it get a context passed in somehow?
UndefinedTensor::UndefinedTensor()
: TensorImpl(&(globalContext().getType(Backend::Undefined,ScalarType::Undefined))) {
}

const char * UndefinedTensor::toString() const {
  return "UndefinedTensor";
}

IntList UndefinedTensor::sizes() const {
  runtime_error("sizes() called on undefined Tensor");
}

int64_t UndefinedTensor::dim() const {
  runtime_error("dim() called on undefined Tensor");
}

const char * UndefinedTensor::typeString() {
  return "UndefinedType";
}
void * UndefinedTensor::unsafeGetTH(bool retain) {
  runtime_error("unsafeGetTH(bool retain) called on undefined Tensor");
}
std::unique_ptr<Storage> UndefinedTensor::storage() {
  runtime_error("storage() called on undefined Tensor");
}

IntList UndefinedTensor::strides() const {
  runtime_error("strides() called on undefined Tensor");
}
Scalar UndefinedTensor::localScalar() {
  runtime_error("localScalar() called on undefined Tensor");
}

UndefinedTensor UndefinedTensor::_singleton;

}
