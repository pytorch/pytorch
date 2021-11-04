#include "lazy_tensor_core/csrc/device.h"

#include <c10/util/Optional.h>
#include "lazy_tensor_core/csrc/compiler/backend_impl_interface.h"
#include "lazy_tensors/str_split.h"

namespace torch_lazy_tensors {
namespace {

thread_local c10::optional<Device> g_current_device;

}  // namespace

Device::Device(const std::string& device_spec) {}

std::string Device::toString() const {
  return c10::str("Default:", ordinal_);
}

int Device::compare(const Device& rhs) const {
  if (type_ != rhs.type_) {
    return type_ < rhs.type_ ? -1 : +1;
  }
  return ordinal_ < rhs.ordinal_ ? -1 : (ordinal_ > rhs.ordinal_ ? +1 : 0);
}

std::ostream& operator<<(std::ostream& os, const Device& device) {
  os << device.toString();
  return os;
}

const Device* GetDefaultDevice() {
  static const Device* default_device = new Device("");
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
