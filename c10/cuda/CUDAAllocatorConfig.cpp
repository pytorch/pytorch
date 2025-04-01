#include <c10/cuda/CUDAAllocatorConfig.h>

namespace c10::cuda::CUDACachingAllocator {

size_t CUDAAllocatorConfig::roundup_power2_divisions(size_t size) {
  return getAllocatorConfig().roundup_power2_divisions(size);
}

void CUDAAllocatorConfig::parseArgs(const char* env) {
  getAllocatorConfig().parseArgs(env);
}

void setAllocatorSettings(const std::string& env) {
  getAllocatorConfig().parseArgs(env.c_str());
}

} // namespace c10::cuda::CUDACachingAllocator
