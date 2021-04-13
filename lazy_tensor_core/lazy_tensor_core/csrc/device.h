#pragma once

#include <iostream>
#include <string>

#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {

enum class DeviceType { CPU, GPU, TPU };

struct Device {
  Device() = default;
  explicit Device(const std::string& device_spec);
  Device(DeviceType hw_type, int ordinal)
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

  size_t hash() const {
    return lazy_tensors::util::StdHashCombine(
        lazy_tensors::util::GetEnumValue(hw_type), ordinal + 1);
  }

  DeviceType hw_type = DeviceType::CPU;
  int ordinal = 0;
};

const Device* GetDefaultDevice();

Device GetCurrentDevice();

Device SetCurrentDevice(const Device& device);

static inline Device GetDeviceOrCurrent(const Device* device) {
  return device != nullptr ? *device : GetCurrentDevice();
}

}  // namespace torch_lazy_tensors
