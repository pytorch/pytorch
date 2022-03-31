#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Runtime.h>
#include <ATen/native/vulkan/api/Shader.h>
#include <iostream>

namespace at {
namespace native {
namespace vulkan {
namespace api {

//
// A Vulkan Adapter represents a logical device and all its properties. It
// manages all relevant properties of the underlying physical device, a 
// handle to the logical device, and a number of compute queues available to
// the device. It is primarily responsible for managing the VkDevice handle
// which points to the logical device object on the GPU.
//
// This class is primarily used by the Runtime class, which holds one Adapter
// instance for each physical device visible to the VkInstance. Upon construction,
// this class will populate the physical device properties, but will not create
// the logical device until specifically requested via the init_device() funtion.
//
// init_device() will create the logical device and obtain the VkDevice handle
// for it. It will also create a number of compute queues up to the amount
// requested when the Adapter instance was constructed.
//
// Contexts (which represent one thread of execution) will request a compute
// queue from an Adapter. The Adapter will then select a compute queue to
// assign to the Context, attempting to balance load between all available
// queues. This will allow different Contexts (which typically execute on
// separate threads) to run concurrently.
//

class Adapter final {
 public:
  explicit Adapter(const VkPhysicalDevice handle, const uint32_t num_queues);

  Adapter(const Adapter&) = delete;
  Adapter& operator=(const Adapter&) = delete;

  Adapter(Adapter&&) noexcept;
  Adapter& operator=(Adapter&&) = delete;

  ~Adapter();

  struct Queue {
    uint32_t family_index;
    uint32_t queue_index;
    VkQueueFlags capabilities;
    VkQueue handle;
  };

 private:
  // Use a mutex to manage resources held by this class since
  // it can be accessed from multiple threads
  std::mutex mutex_;
  // Physical Device Properties
  VkPhysicalDevice physical_handle_;
  VkPhysicalDeviceProperties properties_;
  VkPhysicalDeviceMemoryProperties memory_properties_;
  std::vector<VkQueueFamilyProperties> queue_families_;
  uint32_t compute_queue_family_index_;
  // Queue Management
  uint32_t num_requested_queues_;
  using UsageHeuristic = uint32_t; // In case the UsageHeuristic type needs to be changed later
  std::vector<UsageHeuristic> queue_usage_;
  // Handles
  VkDevice handle_;
  std::vector<Queue> queues_;
  VkQueue queue_;

 public:
  VkPhysicalDevice physical_handle() const;
  uint32_t compute_queue_family_index() const;
  VkDevice device_handle() const;
  VkQueue compute_queue() const;

  void init_device();
  Queue request_queue();
  void return_queue(Queue& compute_queue);

  inline bool has_unified_memory() const {
    // Ideally iterate over all memory types to see if there is a pool that
    // is both host-visible, and device-local.  This should be a good proxy
    // for now.
    return VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU == properties_.deviceType;
  }

  inline Shader::WorkGroup local_work_group_size() const {
    return { 4u, 4u, 4u, };
  }

  friend std::ostream& operator<<(std::ostream& os, const Adapter& adapter);

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
