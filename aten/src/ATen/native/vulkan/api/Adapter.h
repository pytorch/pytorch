#pragma once

#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Runtime.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

struct Adapter final {
  Runtime* runtime;
  VkPhysicalDevice handle;
  VkPhysicalDeviceProperties properties;
  VkPhysicalDeviceMemoryProperties memory_properties;
  uint32_t compute_queue_family_index;

  inline bool has_unified_memory() const {
    return VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU == properties.deviceType;
  }
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
