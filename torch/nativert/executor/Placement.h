#pragma once

#include <c10/core/Device.h>
#include <c10/util/Logging.h>

#include <optional>
#include <unordered_map>

namespace torch::nativert {

/**
 * Returns true if the two devices are the same and has the same device index
 * (if cuda).
 */
bool isSameDevice(const c10::Device& device1, const c10::Device& device2);

/**
 * @brief A utility class for managing device placement mappings.
 *
 * The Placement class provides a way to map source devices to target devices.
 * It supports both explicit per-device mappings and a default device fallback.
 * This is the argument taken in NativeRT to map from model artifact device to
 * the device it should run on.
 */
struct TORCH_API Placement {
  Placement() = default;
  explicit Placement(std::optional<c10::Device> defaultDevice);
  explicit Placement(
      const std::unordered_map<c10::Device, c10::Device>& deviceMap,
      std::optional<c10::Device> defaultDevice = std::nullopt);
  c10::Device getMappedDevice(const c10::Device& srcDevice) const;

  TORCH_API friend std::ostream& operator<<(
      std::ostream& os,
      const Placement& obj);

 protected:
  std::unordered_map<c10::Device, c10::Device> deviceMap_;
  std::optional<c10::Device> defaultDevice_;
};

} // namespace torch::nativert
