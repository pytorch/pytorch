#pragma once

#include <ATen/core/Tensor.h>

namespace lazy_tensors {

class NNCComputationClient {
 public:
  static at::DeviceType HardwareDeviceType();
};

}  // namespace lazy_tensors
