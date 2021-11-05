#include "lazy_tensor_core/csrc/device.h"

#include <c10/util/Logging.h>
#include <c10/util/Optional.h>

namespace torch_lazy_tensors {
namespace {

thread_local c10::optional<Device> g_current_device;

}  // namespace

Device::Device()
  : type_(std::make_shared<BackendDeviceType>()) {}

Device::Device(std::shared_ptr<BackendDeviceType>&& type, int ordinal)
      : type_(std::move(type)), ordinal_(ordinal) {}

Device::Device(const std::string& device_spec) {}

int8_t Device::type() const {
  TORCH_INTERNAL_ASSERT(type_);
  return type_->type;
}

std::string Device::toString() const {
  TORCH_INTERNAL_ASSERT(type_);
  return c10::str(type_->toString(), ordinal_);
}

int Device::compare(const Device& rhs) const {
  if (type() != rhs.type()) {
    return type() < rhs.type() ? -1 : +1;
  }
  return ordinal_ < rhs.ordinal_ ? -1 : (ordinal_ > rhs.ordinal_ ? +1 : 0);
}

std::ostream& operator<<(std::ostream& os, const Device& device) {
  os << device.toString();
  return os;
}

const Device* GetDefaultDevice() {
  static const Device* default_device = new Device();
  return default_device;
}

Device GetCurrentDevice() {
  if (!g_current_device) {
    g_current_device = *GetDefaultDevice();
  }
  return *g_current_device;
}

Device SetCurrentDevice(const Device& device) {
  Device current = GetCurrentDevice();
  g_current_device = device;
  VLOG(2) << "New current device: " << device;
  return current;
}

}  // namespace torch_lazy_tensors
