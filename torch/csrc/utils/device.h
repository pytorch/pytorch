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
  inline int64_t deviceInt64() { return (this->is_default || this->type == DeviceType::CPU) ? -1 : this->index; }
};

}
