#pragma once

#include <ATen/native/vulkan/api/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

struct Adapter final {
  VkInstance instance;
  VkPhysicalDevice physical_device;
  VkPhysicalDeviceProperties physical_device_properties;
  VkPhysicalDeviceMemoryProperties physical_device_memory_properties;
  uint32_t compute_queue_family_index;

  inline bool is_unified_memory_architecture() const {
    return VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ==
        physical_device_properties.deviceType;
  }
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
