#include <torch/csrc/jit/fuser/common/management.h>

#include <c10/core/Device.h>

namespace torch {
namespace jit {
namespace fuser {


// TODO: can likely do better than global fusion counter
// and fusion key -> device map
static int fusion_counter = 0;
static std::unordered_map<int, c10::DeviceType> fusion_to_device_map;

std::unordered_map<int, c10::DeviceType> getFusionToDeviceMap() {
  return fusion_to_device_map;
}

int getAndIncrementGlobalFusionCounter() {
  return fusion_counter++;
}

}}} // torch::jit::fuser
