#pragma once

#include <ATen/ATen.h>
#include "DynamicTypes.h"
#include "device.h"

namespace torch { namespace utils {

Device getDevice(const at::Tensor tensor) {
  return torch::Device(torch::getDeviceType(tensor.type()), tensor.type().is_cuda() ? tensor.get_device(): 0, false);
}

}} // namespace torch::utils
