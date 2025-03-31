#include <c10/cuda/CUDAAllocatorConfig.h>

namespace c10::cuda::CUDACachingAllocator {

size_t CUDAAllocatorConfig::roundup_power2_divisions(size_t size) {
  return c10::CachingAllocator::AllocatorConfig::instance()
      .roundup_power2_divisions(size);
}

void CUDAAllocatorConfig::parseArgs(const char* env) {
  c10::CachingAllocator::AllocatorConfig::instance().parseArgs(env);
}

void setAllocatorSettings(const std::string& env) {
  c10::CachingAllocator::AllocatorConfig::instance().parseArgs(env.c_str());
}

} // namespace c10::cuda::CUDACachingAllocator
