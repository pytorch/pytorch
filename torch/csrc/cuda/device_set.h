#pragma once

#include <bitset>
#include <cstddef>
#include <c10/cuda/CUDAMacros.h>

namespace torch {

using device_set = std::bitset<C10_COMPILE_TIME_MAX_GPUS>;

} // namespace torch
