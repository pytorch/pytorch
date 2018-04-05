#include "device.h"
#include <stdexcept>

namespace torch {

Device::Device(DeviceType device_type, int64_t device_index, bool is_default)
    : device_type(device_type), device_index(device_index), is_default(is_default) {
  if (!is_default) {
    switch (device_type) {
      case DeviceType::CPU:
        if (device_index != 0) {
          throw std::runtime_error("cpu device index must be 0, got " + std::to_string(device_index));
        }
        break;
      case DeviceType::CUDA:
        if (device_index < 0) {
          throw std::runtime_error("device index must be positive, got " + std::to_string(device_index));
        }
        break;
      default:
        throw std::runtime_error("unexpected DeviceType");
    }
  }
}

}
