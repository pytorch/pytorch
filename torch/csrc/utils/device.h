#pragma once

#include <cstdint>

namespace torch {

enum class DeviceType {CPU=0, CUDA=1};

struct Device {
  const DeviceType device_type;
  const int64_t device_index;
  const bool is_default;   // is default device for type.
  Device(DeviceType device_type, int64_t device_index, bool is_default);
};

}
