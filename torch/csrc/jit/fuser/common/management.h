#pragma once

#include <ATen/ATen.h>

#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {

TORCH_API std::unordered_map<int, c10::DeviceType> getFusionToDeviceMap();

TORCH_API int getAndIncrementGlobalFusionCounter();

}}} // torch::jit::fuser
