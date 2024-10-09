#include <torch/csrc/lazy/backend/backend_device.h>

#include <c10/core/Device.h>
#include <c10/util/Exception.h>
#include <c10/util/StringUtil.h>
#include <torch/csrc/lazy/backend/backend_interface.h>
#include <torch/csrc/lazy/core/tensor.h>
#include <optional>

namespace torch::lazy {

BackendDevice::BackendDevice()
    : type_(getBackend()->GetDefaultDeviceType()),
      ordinal_(getBackend()->GetDefaultDeviceOrdinal()) {}

BackendDevice::BackendDevice(
    std::shared_ptr<BackendDeviceType>&& type,
    int64_t ordinal)
    : type_(std::move(type)), ordinal_(ordinal) {}

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

BackendDevice atenDeviceToBackendDevice(const c10::Device& device) {
  TORCH_CHECK(device.type() == at::kLazy, device);
  int64_t ordinal = device.has_index()
      ? device.index()
      : getBackend()->GetDefaultDeviceOrdinal();
  return BackendDevice(getBackend()->GetDefaultDeviceType(), ordinal);
}

// TODO(whc) refactor this: we need to support non 1 on 1 mapping for torch/XLA.
c10::Device backendDeviceToAtenDevice(const BackendDevice& device) {
  return c10::Device(
      at::kLazy, static_cast<c10::DeviceIndex>(device.ordinal()));
}

std::optional<BackendDevice> GetBackendDevice(at::ITensorListRef tensors) {
  for (auto& tensor : tensors) {
    if (auto lt = TryGetLtcTensor(tensor)) {
      return lt->GetDevice();
    }
  }
  return std::nullopt;
}

std::optional<BackendDevice> GetBackendDevice(at::TensorList tensors) {
  return GetBackendDevice(at::ITensorListRef(tensors));
}

std::optional<BackendDevice> GetBackendDevice(const at::Tensor& tensor) {
  if (auto lt = TryGetLtcTensor(tensor)) {
    return lt->GetDevice();
  }
  return std::nullopt;
}

std::optional<BackendDevice> GetBackendDevice(
    const std::optional<c10::Device>& device) {
  if (device) {
    return std::make_optional(atenDeviceToBackendDevice(*device));
  }
  return std::nullopt;
}

std::optional<BackendDevice> GetBackendDevice() {
  return std::nullopt;
}

} // namespace torch::lazy
