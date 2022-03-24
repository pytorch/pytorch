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

class Adapter final {
 public:
  explicit Adapter(const VkPhysicalDevice handle);

  Adapter(const Adapter&) = delete;
  Adapter& operator=(const Adapter&) = delete;

  Adapter(Adapter&&);
  Adapter& operator=(Adapter&&) = delete;

  ~Adapter();

  void init_device();

  VkPhysicalDevice physical_handle() const;
  uint32_t compute_queue_family_index() const;
  VkDevice device_handle() const;
  VkQueue compute_queue() const;

 private:
  VkPhysicalDevice physical_handle_;
  VkPhysicalDeviceProperties properties_;
  VkPhysicalDeviceMemoryProperties memory_properties_;
  std::vector<VkQueueFamilyProperties> queue_families_;
  uint32_t compute_queue_family_index_;

  VkDevice handle_;
  VkQueue queue_;

 public:
  inline bool has_unified_memory() const {
    // Ideally iterate over all memory types to see if there is a pool that
    // is both host-visible, and device-local.  This should be a good proxy
    // for now.
    return VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU == properties_.deviceType;
  }

  inline Shader::WorkGroup local_work_group_size() const {
    return { 4u, 4u, 4u, };
  }
};

//
// Impl
//

inline VkPhysicalDevice Adapter::physical_handle() const {
  return physical_handle_;
}

inline uint32_t Adapter::compute_queue_family_index() const {
  return compute_queue_family_index_;
}

inline VkDevice Adapter::device_handle() const {
  return handle_;
}

inline VkQueue Adapter::compute_queue() const {
  return queue_;
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
