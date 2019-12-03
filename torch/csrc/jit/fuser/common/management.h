#pragma once

namespace torch {
namespace jit {
namespace fuser {


// TODO: can likely do better than global fusion counter
// and fusion key -> device map
int fusion_counter = 0;
std::unordered_map<int, c10::DeviceType> fusion_to_device_map;

TORCH_API std::unordered_map<int, c10::DeviceType> getFusionToDeviceMap() {
  return fusion_to_device_map;
}

TORCH_API int getAndIncrementGlobalFusionCounter() {
  return fusion_counter++;
}

}}} // torch::jit::fuser
