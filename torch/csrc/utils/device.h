#pragma once

#include <cstdint>

namespace torch {

enum class DeviceType {CPU=0, CUDA=1};

struct Device {
  const DeviceType type;
  const int64_t index;
  const bool is_default;   // is default device for type.
  Device(DeviceType type, int64_t index, bool is_default);
};

}
