#include <c10/core/AllocatorConfig.h>
#include <c10/core/DeviceType.h>

#include <array>

namespace c10::CachingAllocator {

static std::array<AllocatorConfig*, at::COMPILE_TIME_MAX_DEVICE_TYPES>
    allocator_configs{};

void SetAllocatorConfig(at::DeviceType t, AllocatorConfig* allocator_config) {
  TORCH_INTERNAL_ASSERT(!allocator_configs[static_cast<int>(t)]);
  allocator_configs[static_cast<int>(t)] = allocator_config;
}

AllocatorConfig* GetAllocatorConfig(const at::DeviceType& t) {
  auto* allocator_config = allocator_configs[static_cast<int>(t)];
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      allocator_config, "AllocatorConfig for ", t, " is not set.");
  return allocator_config;
}

AllocatorConfig& AllocatorConfig::instance() {}

} // namespace c10::CachingAllocator
