#pragma once

#include <iostream>
#include <string>

#include <c10/util/Deprecated.h>

namespace torch_lazy_tensors {

// Backend can define their own enum and mandate that in their implementations.
using HardwareType = uint8_t;

class Device {
 public:
  Device() = default;
  Device(HardwareType type, int ordinal)
      : type_(type), ordinal_(ordinal) {}

  bool operator==(const Device& other) const { return compare(other) == 0; }
  bool operator!=(const Device& other) const { return compare(other) != 0; }
  bool operator<(const Device& rhs) const { return compare(rhs) < 0; }

  std::string toString() const;

  // The string -> Device conversion should be handled by the backend interface.
  C10_DEPRECATED explicit Device(const std::string& device_spec);

 private:
  int compare(const Device& rhs) const;

  HardwareType type_ {0};
  int ordinal_ {0};
};

std::ostream& operator<<(std::ostream& os, const Device& device);

const Device* GetDefaultDevice();

Device GetCurrentDevice();

Device SetCurrentDevice(const Device& device);

static inline Device GetDeviceOrCurrent(const Device* device) {
  return device != nullptr ? *device : GetCurrentDevice();
}

}  // namespace torch_lazy_tensors
