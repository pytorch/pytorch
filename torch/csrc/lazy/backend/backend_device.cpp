#include <torch/csrc/lazy/backend/backend_device.h>

#include <c10/util/Exception.h>
#include <c10/util/StringUtil.h>

namespace torch {
namespace lazy {

// TODO(alanwaketan): Use the backend API to get the default device type.
// In the future, we should also get the default device ordinal.
BackendDevice::BackendDevice()
  : type_(std::make_shared<BackendDeviceType>()) {}

BackendDevice::BackendDevice(std::shared_ptr<BackendDeviceType>&& type, int ordinal)
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

}  // namespace lazy
}  // namespace torch
