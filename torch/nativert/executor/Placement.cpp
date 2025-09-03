#include <torch/nativert/executor/Placement.h>

#include <fmt/ostream.h>
#include <ostream>

namespace torch::nativert {

std::ostream& operator<<(std::ostream& os, const Placement& placement) {
  std::vector<std::pair<std::string, c10::Device>> sorted_keys;
  sorted_keys.reserve(placement.deviceMap_.size());
  for (const auto& pair : placement.deviceMap_) {
    sorted_keys.emplace_back(pair.first.str(), pair.first);
  }
  std::sort(
      sorted_keys.begin(), sorted_keys.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
      });

  bool first = true;
  for (const auto& pair : sorted_keys) {
    if (!first) {
      fmt::print(os, ",");
    }
    first = false;
    const auto& key = pair.second;
    const auto& value = placement.deviceMap_.at(key);
    fmt::print(os, "{}|{}", pair.first, value.str());
  }
  if (placement.defaultDevice_.has_value()) {
    fmt::print(os, "{}|{}", first ? "" : ",", placement.defaultDevice_->str());
  }
  return os;
}

namespace {
void assertCudaDeviceHasIndex(const c10::Device& device) {
  if (device.is_cuda()) {
    TORCH_CHECK(
        device.has_index(), "CUDA device in placement must have an index");
  }
}
} // namespace

Placement::Placement(std::optional<c10::Device> defaultDevice)
    : Placement({}, defaultDevice) {}

Placement::Placement(
    const std::unordered_map<c10::Device, c10::Device>& deviceMap,
    std::optional<c10::Device> defaultDevice) {
  for (const auto& [srcDevice, dstDevice] : deviceMap) {
    assertCudaDeviceHasIndex(srcDevice);
    assertCudaDeviceHasIndex(dstDevice);

    deviceMap_.try_emplace(srcDevice, dstDevice);
  }

  if (defaultDevice.has_value()) {
    assertCudaDeviceHasIndex(defaultDevice.value());
    defaultDevice_ = defaultDevice.value();
  }
}

c10::Device Placement::getMappedDevice(const c10::Device& srcDevice) const {
  auto it = deviceMap_.find(srcDevice);
  if (it != deviceMap_.end()) {
    return it->second;
  }
  if (defaultDevice_.has_value()) {
    return defaultDevice_.value();
  }
  return srcDevice;
}

} // namespace torch::nativert
