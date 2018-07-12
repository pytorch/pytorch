#include "caffe2/core/dispatch/TensorTypeIdRegistration.h"
#include "caffe2/utils/C++17.h"

namespace c10 {

constexpr TensorTypeId TensorTypeIdCreator::max_id_;

TensorTypeIds::TensorTypeIds()
: creator_(), registry_() {}

TensorTypeIds& TensorTypeIds::singleton() {
  static TensorTypeIds singleton;
  return singleton;
}

TensorTypeIdCreator::TensorTypeIdCreator()
: last_id_(0) {}

TensorTypeId TensorTypeIdCreator::create() {
  auto id = TensorTypeId(++last_id_);

  if (id == max_id_) {
    // If this happens in prod, we have to change details::_tensorTypeId_underlyingType to uint16_t.
    throw std::logic_error("Tried to define more than " + c10::guts::to_string(std::numeric_limits<details::_tensorTypeId_underlyingType>::max()-1) + " tensor types, which is unsupported");
  }

  return id;
}

TensorTypeIdRegistry::TensorTypeIdRegistry()
: registeredTypeIds_(), mutex_() {}

void TensorTypeIdRegistry::registerId(TensorTypeId id) {
  std::lock_guard<std::mutex> lock(mutex_);
  registeredTypeIds_.emplace(id);
}

void TensorTypeIdRegistry::deregisterId(TensorTypeId id) {
  std::lock_guard<std::mutex> lock(mutex_);
  registeredTypeIds_.erase(id);
}

TensorTypeId TensorTypeIds::createAndRegister() {
  TensorTypeId id = creator_.create();
  registry_.registerId(id);
  return id;
}

void TensorTypeIds::deregister(TensorTypeId id) {
  registry_.deregisterId(id);
}

TensorTypeIdRegistrar::TensorTypeIdRegistrar()
: id_(TensorTypeIds::singleton().createAndRegister()) {
}

TensorTypeIdRegistrar::~TensorTypeIdRegistrar() {
  TensorTypeIds::singleton().deregister(id_);
}

}  // namespace c10
