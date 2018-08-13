#include <ATen/core/TensorTypeIdRegistration.h>
#include <ATen/core/C++17.h>
#include <ATen/core/Error.h>

namespace at {

TensorTypeIds::TensorTypeIds() : creator_(), registry_() {}

TensorTypeIds& TensorTypeIds::singleton() {
  static TensorTypeIds singleton;
  return singleton;
}

TensorTypeIdCreator::TensorTypeIdCreator() : last_id_(0) {}

at::TensorTypeId TensorTypeIdCreator::create() {

  auto id = TensorTypeId(++last_id_);

  if (last_id_ == 0) { // overflow happened!
    // If this happens in prod, we have to change
    // details::_tensorTypeId_underlyingType to uint16_t.
    AT_ERROR(
        "Tried to define more than ",
        std::numeric_limits<details::_tensorTypeId_underlyingType>::max() - 1,
        " tensor types, which is unsupported");
  }

  return id;
}

TensorTypeIdRegistry::TensorTypeIdRegistry() : registeredTypeIds_(), mutex_() {}

void TensorTypeIdRegistry::registerId(at::TensorTypeId id) {
  std::lock_guard<std::mutex> lock(mutex_);
  registeredTypeIds_.emplace(id);
}

void TensorTypeIdRegistry::deregisterId(at::TensorTypeId id) {
  std::lock_guard<std::mutex> lock(mutex_);
  registeredTypeIds_.erase(id);
}

at::TensorTypeId TensorTypeIds::createAndRegister() {
  at::TensorTypeId id = creator_.create();
  registry_.registerId(id);
  return id;
}

void TensorTypeIds::deregister(at::TensorTypeId id) {
  registry_.deregisterId(id);
}

TensorTypeIdRegistrar::TensorTypeIdRegistrar()
    : id_(TensorTypeIds::singleton().createAndRegister()) {}

TensorTypeIdRegistrar::~TensorTypeIdRegistrar() {
  TensorTypeIds::singleton().deregister(id_);
}

AT_DEFINE_TENSOR_TYPE(UndefinedTensorId);
AT_DEFINE_TENSOR_TYPE(CPUTensorId);
AT_DEFINE_TENSOR_TYPE(CUDATensorId);
AT_DEFINE_TENSOR_TYPE(SparseCPUTensorId);
AT_DEFINE_TENSOR_TYPE(SparseCUDATensorId);

} // namespace at
