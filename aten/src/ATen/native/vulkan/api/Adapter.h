#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Utils.h>
#include <ostream>
#include <iostream>

namespace at {
namespace native {
namespace vulkan {
namespace api {

struct PhysicalDevice final {
  // Handle
  VkPhysicalDevice handle;

  // Properties obtained from Vulkan
  VkPhysicalDeviceProperties properties;
  VkPhysicalDeviceMemoryProperties memory_properties;
  std::vector<VkQueueFamilyProperties> queue_families;

  // Metadata
  uint32_t num_compute_queues;
  bool has_unified_memory;
  bool has_timestamps;
  float timestamp_period;

  explicit PhysicalDevice(const VkPhysicalDevice);
};

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
  explicit Adapter(
      const PhysicalDevice& physical_device, const uint32_t num_queues);

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
  // Physical Device Info
  PhysicalDevice physical_device_;
  // Queue Management
  std::vector<Queue> queues_;
  std::vector<uint32_t> queue_usage_;
  // Handles
  VkDevice handle_;

 public:
  inline VkPhysicalDevice physical_handle() const {
    return physical_device_.handle;
  }

  inline VkDevice device_handle() const {
    return handle_;
  }

  inline bool has_unified_memory() const {
    return physical_device_.has_unified_memory;
  }

  inline uint32_t num_compute_queues() const {
    return physical_device_.num_compute_queues;
  }

  inline bool timestamp_compute_and_graphics() const {
    return physical_device_.has_timestamps;
  }

  inline float timestamp_period() const {
    return physical_device_.timestamp_period;
  }

  Queue request_queue();
  void return_queue(Queue&);

  inline utils::uvec3 local_work_group_size() const {
    return { 4u, 4u, 4u, };
  }

  std::string stringize() const;
  friend std::ostream& operator<<(std::ostream&, const Adapter&);
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
