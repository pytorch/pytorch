#pragma once

#include <iostream>
#include <string>

namespace torch_lazy_tensors {

// Backend can define their own enum and mandate that in their implementations.
using HardwareType = uint8_t;

struct Device {
  Device() = default;
  explicit Device(const std::string& device_spec);
  Device(HardwareType hw_type, int ordinal)
      : hw_type(hw_type), ordinal(ordinal) {}

  bool operator==(const Device& other) const { return compare(other) == 0; }

  bool operator!=(const Device& other) const { return compare(other) != 0; }

  bool operator<(const Device& rhs) const { return compare(rhs) < 0; }

  int compare(const Device& rhs) const {
    if (hw_type != rhs.hw_type) {
      return hw_type < rhs.hw_type ? -1 : +1;
    }
    return ordinal < rhs.ordinal ? -1 : (ordinal > rhs.ordinal ? +1 : 0);
  }

  std::string ToString() const;

  friend std::ostream& operator<<(std::ostream& os, const Device& device) {
    os << device.ToString();
    return os;
  }

  HardwareType hw_type {0};
  int ordinal {0};
};

const Device* GetDefaultDevice();

Device GetCurrentDevice();

Device SetCurrentDevice(const Device& device);

static inline Device GetDeviceOrCurrent(const Device* device) {
  return device != nullptr ? *device : GetCurrentDevice();
}

}  // namespace torch_lazy_tensors
