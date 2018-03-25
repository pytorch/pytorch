#pragma once

#include <bitset>

namespace torch {

static constexpr std::size_t MAX_CUDA_DEVICES = 64;
using device_set = std::bitset<MAX_CUDA_DEVICES>;

}
