#pragma once

#include <c10/core/Device.h>
#include <bitset>
#include <cstddef>

namespace torch {

using device_set = std::bitset<c10::Device::MAX_NUM_DEVICES>;

} // namespace torch
