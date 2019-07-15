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

TensorTypeId TensorTypeIdCreator::create() {
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

void TensorTypeIdRegistry::registerId(TensorTypeId id, std::string name) {
  std::lock_guard<std::mutex> lock(mutex_);
  registeredTypeIds_.emplace(id, std::move(name));
}

void TensorTypeIdRegistry::deregisterId(TensorTypeId id) {
  std::lock_guard<std::mutex> lock(mutex_);
  registeredTypeIds_.erase(id);
}

const std::string& TensorTypeIdRegistry::toString(TensorTypeId id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto found = registeredTypeIds_.find(id);
  TORCH_INTERNAL_ASSERT(found != registeredTypeIds_.end());
  return found->second;
}

TensorTypeId TensorTypeIds::createAndRegister(std::string name) {
  TensorTypeId id = creator_.create();
  registry_.registerId(id, std::move(name));
  return id;
}

void TensorTypeIds::deregister(TensorTypeId id) {
  registry_.deregisterId(id);
}

const std::string& TensorTypeIds::toString(TensorTypeId id) const {
  return registry_.toString(id);
}

TensorTypeIdRegistrar::TensorTypeIdRegistrar(std::string name)
    : id_(TensorTypeIds::singleton().createAndRegister(std::move(name))) {}

TensorTypeIdRegistrar::~TensorTypeIdRegistrar() {
  TensorTypeIds::singleton().deregister(id_);
}

std::ostream& operator<<(std::ostream& str, c10::TensorTypeId rhs) {
  return str << toString(rhs);
}

std::string toString(TensorTypeId id) {
  return TensorTypeIds::singleton().toString(id);
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
C10_DEFINE_TENSOR_TYPE(MkldnnCPUTensorId);
C10_DEFINE_TENSOR_TYPE(QuantizedCPUTensorId);
C10_DEFINE_TENSOR_TYPE(ComplexCPUTensorId);
C10_DEFINE_TENSOR_TYPE(ComplexCUDATensorId);

} // namespace c10
