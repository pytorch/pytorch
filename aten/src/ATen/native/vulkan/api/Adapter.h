#pragma once

#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Runtime.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

struct Adapter final {
  Runtime* runtime;
  VkPhysicalDevice physical_device;
  VkPhysicalDeviceProperties physical_device_properties;
  uint32_t compute_queue_family_index;
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
