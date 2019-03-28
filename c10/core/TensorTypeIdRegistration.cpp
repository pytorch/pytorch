#include <c10/core/TensorTypeIdRegistration.h>
#include <c10/util/C++17.h>
#include <c10/util/Exception.h>

namespace c10 {

TensorTypeIds::TensorTypeIds() : creator_(), registry_() {}

TensorTypeIds& TensorTypeIds::singleton() {
  static TensorTypeIds singleton;
  return singleton;
}

TensorTypeIdCreator::TensorTypeIdCreator() : last_id_(0) {}

c10::TensorTypeId TensorTypeIdCreator::create() {
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

void TensorTypeIdRegistry::registerId(c10::TensorTypeId id) {
  std::lock_guard<std::mutex> lock(mutex_);
  registeredTypeIds_.emplace(id);
}

void TensorTypeIdRegistry::deregisterId(c10::TensorTypeId id) {
  std::lock_guard<std::mutex> lock(mutex_);
  registeredTypeIds_.erase(id);
}

c10::TensorTypeId TensorTypeIds::createAndRegister() {
  c10::TensorTypeId id = creator_.create();
  registry_.registerId(id);
  return id;
}

void TensorTypeIds::deregister(c10::TensorTypeId id) {
  registry_.deregisterId(id);
}

TensorTypeIdRegistrar::TensorTypeIdRegistrar()
    : id_(TensorTypeIds::singleton().createAndRegister()) {}

TensorTypeIdRegistrar::~TensorTypeIdRegistrar() {
  TensorTypeIds::singleton().deregister(id_);
}

C10_DEFINE_TENSOR_TYPE(UndefinedTensorId);
C10_DEFINE_TENSOR_TYPE(CPUTensorId);
C10_DEFINE_TENSOR_TYPE(CUDATensorId);
C10_DEFINE_TENSOR_TYPE(SparseCPUTensorId);
C10_DEFINE_TENSOR_TYPE(SparseCUDATensorId);
C10_DEFINE_TENSOR_TYPE(MKLDNNTensorId);
C10_DEFINE_TENSOR_TYPE(OpenGLTensorId);
C10_DEFINE_TENSOR_TYPE(OpenCLTensorId);
C10_DEFINE_TENSOR_TYPE(IDEEPTensorId);
C10_DEFINE_TENSOR_TYPE(HIPTensorId);
C10_DEFINE_TENSOR_TYPE(SparseHIPTensorId);
C10_DEFINE_TENSOR_TYPE(MSNPUTensorId);
C10_DEFINE_TENSOR_TYPE(XLATensorId);

} // namespace c10
