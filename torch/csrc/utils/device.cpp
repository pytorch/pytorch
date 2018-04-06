#include "device.h"
#include <stdexcept>
#include <string>

namespace torch {

Device::Device(DeviceType type, int64_t index, bool is_default)
    : type(type), index(index), is_default(is_default) {
  if (!is_default) {
    switch (type) {
      case DeviceType::CPU:
        if (index != 0) {
          throw std::runtime_error("cpu device index must be 0, got " + std::to_string(index));
        }
        break;
      case DeviceType::CUDA:
        if (index < 0) {
          throw std::runtime_error("device index must be positive, got " + std::to_string(index));
        }
        break;
      default:
        throw std::runtime_error("unexpected DeviceType");
    }
  }
}

bool Device::operator==(const Device& rhs) {
  return this->type == rhs.type && this->index == rhs.index && this->is_default == rhs.is_default;
}

}
