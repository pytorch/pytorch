#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Runtime.h>
#include <ATen/native/vulkan/api/Shader.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

//
// A Vulkan Adapter represents a physical device and its properties.  Adapters
// are enumerated through the Runtime and are used in creation of Contexts.
// Each tensor in PyTorch is associated with a Context to make the
// device <-> tensor affinity explicit.
//

struct Adapter final {
  Runtime* runtime;
  VkPhysicalDevice handle;
  VkPhysicalDeviceProperties properties;
  VkPhysicalDeviceMemoryProperties memory_properties;
  uint32_t compute_queue_family_index;

  inline bool has_unified_memory() const {
    // Ideally iterate over all memory types to see if there is a pool that
    // is both host-visible, and device-local.  This should be a good proxy
    // for now.
    return VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU == properties.deviceType;
  }

  inline Shader::WorkGroup local_work_group_size() const {
    return { 4u, 4u, 4u, };
  }
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
