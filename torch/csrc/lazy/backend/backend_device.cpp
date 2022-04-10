#include <torch/csrc/lazy/backend/backend_device.h>

#include <c10/core/Device.h>
#include <c10/util/Exception.h>
#include <c10/util/StringUtil.h>
#include <torch/csrc/lazy/core/tensor.h>
#include <torch/csrc/lazy/backend/backend_interface.h>

namespace torch {
namespace lazy {

// TODO(alanwaketan): Use the backend API to get the default device type.
// In the future, we should also get the default device ordinal.
BackendDevice::BackendDevice()
  : type_(std::make_shared<BackendDeviceType>()) {}

BackendDevice::BackendDevice(std::shared_ptr<BackendDeviceType>&& type, int64_t ordinal)
  : type_(std::move(type)), ordinal_(ordinal) {}

BackendDevice::BackendDevice(const std::string& device_spec)
  : BackendDevice::BackendDevice() {}

int8_t BackendDevice::type() const {
  TORCH_INTERNAL_ASSERT(type_);
  return type_->type;
}

std::string BackendDevice::toString() const {
  TORCH_INTERNAL_ASSERT(type_);
  return c10::str(type_->toString(), ordinal_);
}

int BackendDevice::compare(const BackendDevice& rhs) const {
  if (type() != rhs.type()) {
    return type() < rhs.type() ? -1 : +1;
  }
  return ordinal_ < rhs.ordinal_ ? -1 : (ordinal_ > rhs.ordinal_ ? +1 : 0);
}

std::ostream& operator<<(std::ostream& os, const BackendDevice& device) {
  os << device.toString();
  return os;
}

// TODO(whc) refactor this: we need to support non-zero default ordinal for torch/XLA.
BackendDevice atenDeviceToBackendDevice(const c10::Device& device) {
  TORCH_CHECK(device.type() == at::kLazy, device);
  int64_t ordinal = device.has_index() ? device.index() : 0;
  return BackendDevice(getBackend()->GetDefaultDeviceType(), ordinal);
}

// TODO(whc) refactor this: we need to support non 1 on 1 mapping for torch/XLA.
c10::Device backendDeviceToAtenDevice(const BackendDevice& device) {
  return c10::Device(at::kLazy, device.ordinal());
}

c10::optional<BackendDevice> GetBackendDevice(const at::TensorList tensors) {
  for (auto& tensor: tensors) {
    if (auto lt = TryGetLtcTensor(tensor)) {
      return lt->GetDevice();
    }
  }
  return c10::nullopt;
}

c10::optional<BackendDevice> GetBackendDevice(const at::Tensor& tensor) {
  if (auto lt = TryGetLtcTensor(tensor)) {
    return lt->GetDevice();
  }
  return c10::nullopt;
}

c10::optional<BackendDevice> GetBackendDevice() {
  return c10::nullopt;
}

}  // namespace lazy
}  // namespace torch
