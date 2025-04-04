#include "torch/csrc/nativert/executor/Placement.h"

#include <map>

namespace torch::nativert {

std::ostream& operator<<(std::ostream& os, const Placement& placement) {
  std::map<std::string, c10::Device> keys;
  for (const auto& pair : placement.deviceMap_) {
    keys.insert({pair.first.str(), pair.first});
  }

  bool first = true;
  auto checkComma = [&]() {
    if (!first) {
      os << ",";
    }
    first = false;
  };

  os << "";
  for (const auto& pair : keys) {
    checkComma();
    const auto& key = pair.second;
    const auto& value = placement.deviceMap_.at(key);
    os << pair.first << "|" << value.str();
  }

  if (placement.defaultDevice_.has_value()) {
    checkComma();
    os << "|" << placement.defaultDevice_.value().str();
  }
  return os;
}

c10::Device normalizeDevice(const c10::Device& device) {
  // cpu device doesn't have index
  // cuda device index must have a index
  if (device.is_cpu()) {
    return c10::Device(c10::DeviceType::CPU);
  } else if (device.is_cuda()) {
    return c10::Device(
        c10::DeviceType::CUDA, device.has_index() ? device.index() : 0);
  } else {
    TORCH_CHECK(false, "Unsupported device type", device);
  }
}

bool isSameDevice(const c10::Device& a, const c10::Device& b) {
  if (a.is_cpu()) {
    return b.is_cpu();
  }
  if (a.is_cuda()) {
    if (b.is_cuda()) {
      auto aIndex = a.has_index() ? a.index() : 0;
      auto bIndex = b.has_index() ? b.index() : 0;
      return aIndex == bIndex;
    } else {
      return false;
    }
  }
  TORCH_CHECK(false, "Unsupported device type", a, " and ", b);
  return false;
}

Placement::Placement(std::optional<c10::Device> defaultDevice)
    : Placement({}, defaultDevice) {}

Placement::Placement(
    const std::unordered_map<c10::Device, c10::Device>& deviceMap,
    std::optional<c10::Device> defaultDevice) {
  for (const auto& [srcDevice, dstDevice] : deviceMap) {
    deviceMap_.emplace(normalizeDevice(srcDevice), normalizeDevice(dstDevice));
  }
  if (defaultDevice.has_value()) {
    defaultDevice_ = normalizeDevice(defaultDevice.value());
  }
}

c10::Device Placement::getMappedDevice(const c10::Device& srcDevice) const {
  auto it = deviceMap_.find(normalizeDevice(srcDevice));
  if (it != deviceMap_.end()) {
    return it->second;
  }
  if (defaultDevice_.has_value()) {
    return defaultDevice_.value();
  }
  return srcDevice;
}

} // namespace torch::nativert
