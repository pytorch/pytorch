#pragma once

#include <cstdint>

namespace torch {

enum class DeviceType {CPU=0, CUDA=1};

struct Device {
  DeviceType type;
  int64_t index;
  bool is_default;   // is default device for type.
  Device(DeviceType type, int64_t index, bool is_default);
  bool operator==(const Device& rhs);
};

}
