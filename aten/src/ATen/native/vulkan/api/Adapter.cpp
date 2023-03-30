#include <ATen/native/vulkan/api/Adapter.h>
#include <c10/util/irange.h>

#include <bitset>
#include <iomanip>
#include <sstream>

namespace at {
namespace native {
namespace vulkan {
namespace api {

PhysicalDevice::PhysicalDevice(const VkPhysicalDevice physical_device_handle)
    : handle(physical_device_handle),
      properties{},
      memory_properties{},
      queue_families{},
      num_compute_queues(0),
      has_unified_memory(false),
      has_timestamps(false),
      timestamp_period(0) {
  // Extract physical device properties
  vkGetPhysicalDeviceProperties(handle, &properties);
  vkGetPhysicalDeviceMemoryProperties(handle, &memory_properties);

  has_timestamps = properties.limits.timestampComputeAndGraphics;
  timestamp_period = properties.limits.timestampPeriod;

  // Check if there are any memory types have both the HOST_VISIBLE and the
  // DEVICE_LOCAL property flags
  const VkMemoryPropertyFlags unified_memory_flags =
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
  for (const uint32_t i : c10::irange(memory_properties.memoryTypeCount)) {
    if (memory_properties.memoryTypes[i].propertyFlags | unified_memory_flags) {
      has_unified_memory = true;
      break;
    }
  }

  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(
      handle, &queue_family_count, nullptr);

  queue_families.resize(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(
      handle, &queue_family_count, queue_families.data());

  // Find the total number of compute queues
  for (const uint32_t family_i : c10::irange(queue_families.size())) {
    const VkQueueFamilyProperties& properties = queue_families[family_i];
    // Check if this family has compute capability
    if (properties.queueFlags & VK_QUEUE_COMPUTE_BIT) {
      num_compute_queues += properties.queueCount;
    }
  }
}

namespace {

void find_requested_device_extensions(
    VkPhysicalDevice physical_device,
    std::vector<const char*>& enabled_extensions,
    const std::vector<const char*>& requested_extensions) {
  uint32_t device_extension_properties_count = 0;
  VK_CHECK(vkEnumerateDeviceExtensionProperties(
      physical_device, nullptr, &device_extension_properties_count, nullptr));
  std::vector<VkExtensionProperties> device_extension_properties(
      device_extension_properties_count);
  VK_CHECK(vkEnumerateDeviceExtensionProperties(
      physical_device,
      nullptr,
      &device_extension_properties_count,
      device_extension_properties.data()));

  std::vector<const char*> enabled_device_extensions;

  for (const auto& requested_extension : requested_extensions) {
    for (const auto& extension : device_extension_properties) {
      if (strcmp(requested_extension, extension.extensionName) == 0) {
        enabled_extensions.push_back(requested_extension);
        break;
      }
    }
  }
}

VkDevice create_logical_device(
    const PhysicalDevice& physical_device,
    const uint32_t num_queues_to_create,
    std::vector<Adapter::Queue>& queues,
    std::vector<uint32_t>& queue_usage) {
  // Find compute queues up to the requested number of queues

  std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
  queue_create_infos.reserve(num_queues_to_create);

  std::vector<std::pair<uint32_t, uint32_t>> queues_to_get;
  queues_to_get.reserve(num_queues_to_create);

  uint32_t remaining_queues = num_queues_to_create;
  for (const uint32_t family_i :
       c10::irange(physical_device.queue_families.size())) {
    const VkQueueFamilyProperties& queue_properties =
        physical_device.queue_families[family_i];
    // Check if this family has compute capability
    if (queue_properties.queueFlags & VK_QUEUE_COMPUTE_BIT) {
      const uint32_t queues_to_init =
          std::min(remaining_queues, queue_properties.queueCount);

      const std::vector<float> queue_priorities(queues_to_init, 1.0f);
      queue_create_infos.push_back({
          VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, // sType
          nullptr, // pNext
          0u, // flags
          family_i, // queueFamilyIndex
          queues_to_init, // queueCount
          queue_priorities.data(), // pQueuePriorities
      });

      for (const uint32_t queue_i : c10::irange(queues_to_init)) {
        // Use this to get the queue handle once device is created
        queues_to_get.emplace_back(family_i, queue_i);
      }
      remaining_queues -= queues_to_init;
    }
    if (remaining_queues == 0) {
      break;
    }
  }

  queues.reserve(queues_to_get.size());
  queue_usage.reserve(queues_to_get.size());

  // Create the VkDevice

  std::vector<const char*> requested_device_extensions{
#ifdef VK_KHR_portability_subset
      VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME,
#endif /* VK_KHR_portability_subset */
  };

  std::vector<const char*> enabled_device_extensions;
  find_requested_device_extensions(
      physical_device.handle,
      enabled_device_extensions,
      requested_device_extensions);

  const VkDeviceCreateInfo device_create_info{
      VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      static_cast<uint32_t>(queue_create_infos.size()), // queueCreateInfoCount
      queue_create_infos.data(), // pQueueCreateInfos
      0u, // enabledLayerCount
      nullptr, // ppEnabledLayerNames
      static_cast<uint32_t>(
          enabled_device_extensions.size()), // enabledExtensionCount
      enabled_device_extensions.data(), // ppEnabledExtensionNames
      nullptr, // pEnabledFeatures
  };

  VkDevice handle;
  VK_CHECK(vkCreateDevice(
      physical_device.handle, &device_create_info, nullptr, &handle));

#ifdef USE_VULKAN_VOLK
  volkLoadDevice(handle);
#endif /* USE_VULKAN_VOLK */

  // Obtain handles for the created queues and initialize queue usage heuristic

  for (const std::pair<uint32_t, uint32_t>& queue_idx : queues_to_get) {
    VkQueue queue_handle = VK_NULL_HANDLE;
    VkQueueFlags flags =
        physical_device.queue_families[queue_idx.first].queueFlags;
    vkGetDeviceQueue(handle, queue_idx.first, queue_idx.second, &queue_handle);
    queues.push_back({queue_idx.first, queue_idx.second, flags, queue_handle});
    // Initial usage value
    queue_usage.push_back(0);
  }

  return handle;
}

// Print utils

std::string get_device_type_str(const VkPhysicalDeviceType type) {
  switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
      return "INTEGRATED_GPU";
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
      return "DISCRETE_GPU";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
      return "VIRTUAL_GPU";
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
      return "CPU";
    default:
      return "UNKNOWN";
  }
}

std::string get_memory_properties_str(const VkMemoryPropertyFlags flags) {
  std::bitset<10> values(flags);
  std::stringstream ss("|");
  if (values[0]) {
    ss << " DEVICE_LOCAL |";
  }
  if (values[1]) {
    ss << " HOST_VISIBLE |";
  }
  if (values[2]) {
    ss << " HOST_COHERENT |";
  }
  if (values[3]) {
    ss << " HOST_CACHED |";
  }
  if (values[4]) {
    ss << " LAZILY_ALLOCATED |";
  }

  return ss.str();
}

std::string get_queue_family_properties_str(const VkQueueFlags flags) {
  std::bitset<10> values(flags);
  std::stringstream ss("|");
  if (values[0]) {
    ss << " GRAPHICS |";
  }
  if (values[1]) {
    ss << " COMPUTE |";
  }
  if (values[2]) {
    ss << " TRANSFER |";
  }

  return ss.str();
}

} // namespace

//
// DeviceHandle
//

DeviceHandle::DeviceHandle(const VkDevice device) : handle_(device) {}

DeviceHandle::DeviceHandle(DeviceHandle&& other) noexcept
    : handle_(other.handle_) {
  other.handle_ = VK_NULL_HANDLE;
}

DeviceHandle::~DeviceHandle() {
  if C10_LIKELY (VK_NULL_HANDLE == handle_) {
    return;
  }
  vkDestroyDevice(handle_, nullptr);
}

//
// Adapter
//

Adapter::Adapter(
    const VkInstance instance,
    const PhysicalDevice& physical_device,
    const uint32_t num_queues)
    : queue_usage_mutex_{},
      physical_device_(physical_device),
      queues_{},
      queue_usage_{},
      queue_mutexes_{},
      instance_(instance),
      device_(create_logical_device(
          physical_device_,
          num_queues,
          queues_,
          queue_usage_)),
      shader_layout_cache_(device_.handle_),
      shader_cache_(device_.handle_),
      pipeline_layout_cache_(device_.handle_),
      compute_pipeline_cache_(device_.handle_),
      sampler_cache_(device_.handle_),
      vma_(instance_, physical_device_.handle, device_.handle_) {}

Adapter::Queue Adapter::request_queue() {
  // Lock the mutex as multiple threads can request a queue at the same time
  std::lock_guard<std::mutex> lock(queue_usage_mutex_);

  uint32_t min_usage = UINT32_MAX;
  uint32_t min_used_i = 0;
  for (const uint32_t i : c10::irange(queues_.size())) {
    if (queue_usage_[i] < min_usage) {
      min_used_i = i;
      min_usage = queue_usage_[i];
    }
  }
  queue_usage_[min_used_i] += 1;

  return queues_[min_used_i];
}

void Adapter::return_queue(Adapter::Queue& compute_queue) {
  for (const uint32_t i : c10::irange(queues_.size())) {
    if ((queues_[i].family_index == compute_queue.family_index) &&
        (queues_[i].queue_index == compute_queue.queue_index)) {
      std::lock_guard<std::mutex> lock(queue_usage_mutex_);
      queue_usage_[i] -= 1;
      break;
    }
  }
}

void Adapter::submit_cmd(
    const Adapter::Queue& device_queue,
    const VkCommandBuffer cmd,
    const VkFence fence) {
  const VkSubmitInfo submit_info{
      VK_STRUCTURE_TYPE_SUBMIT_INFO, // sType
      nullptr, // pNext
      0u, // waitSemaphoreCount
      nullptr, // pWaitSemaphores
      nullptr, // pWaitDstStageMask
      1u, // commandBufferCount
      &cmd, // pCommandBuffers
      0u, // signalSemaphoreCount
      nullptr, // pSignalSemaphores
  };

  std::lock_guard<std::mutex> queue_lock(
      queue_mutexes_[device_queue.queue_index % NUM_QUEUE_MUTEXES]);

  VK_CHECK(vkQueueSubmit(device_queue.handle, 1u, &submit_info, fence));
}

void Adapter::submit_cmds(
    const Adapter::Queue& device_queue,
    const std::vector<VkCommandBuffer>& cmds,
    const VkFence fence) {
  const VkSubmitInfo submit_info{
      VK_STRUCTURE_TYPE_SUBMIT_INFO, // sType
      nullptr, // pNext
      0u, // waitSemaphoreCount
      nullptr, // pWaitSemaphores
      nullptr, // pWaitDstStageMask
      utils::safe_downcast<uint32_t>(cmds.size()), // commandBufferCount
      cmds.data(), // pCommandBuffers
      0u, // signalSemaphoreCount
      nullptr, // pSignalSemaphores
  };

  VK_CHECK(vkQueueSubmit(device_queue.handle, 1u, &submit_info, fence));
}

std::string Adapter::stringize() const {
  std::stringstream ss;

  VkPhysicalDeviceProperties properties = physical_device_.properties;
  uint32_t v_major = VK_VERSION_MAJOR(properties.apiVersion);
  uint32_t v_minor = VK_VERSION_MINOR(properties.apiVersion);
  std::string device_type = get_device_type_str(properties.deviceType);
  VkPhysicalDeviceLimits limits = properties.limits;

  ss << "{" << std::endl;
  ss << "  Physical Device Info {" << std::endl;
  ss << "    apiVersion:    " << v_major << "." << v_minor << std::endl;
  ss << "    driverversion: " << properties.driverVersion << std::endl;
  ss << "    deviceType:    " << device_type << std::endl;
  ss << "    deviceName:    " << properties.deviceName << std::endl;

#define PRINT_LIMIT_PROP(name)                                         \
  ss << "      " << std::left << std::setw(36) << #name << limits.name \
     << std::endl;

#define PRINT_LIMIT_PROP_VEC3(name)                                       \
  ss << "      " << std::left << std::setw(36) << #name << limits.name[0] \
     << "," << limits.name[1] << "," << limits.name[2] << std::endl;

  ss << "    Physical Device Limits {" << std::endl;
  PRINT_LIMIT_PROP(maxImageDimension1D);
  PRINT_LIMIT_PROP(maxImageDimension2D);
  PRINT_LIMIT_PROP(maxImageDimension3D);
  PRINT_LIMIT_PROP(maxTexelBufferElements);
  PRINT_LIMIT_PROP(maxPushConstantsSize);
  PRINT_LIMIT_PROP(maxMemoryAllocationCount);
  PRINT_LIMIT_PROP(maxSamplerAllocationCount);
  PRINT_LIMIT_PROP(maxComputeSharedMemorySize);
  PRINT_LIMIT_PROP_VEC3(maxComputeWorkGroupCount);
  PRINT_LIMIT_PROP(maxComputeWorkGroupInvocations);
  PRINT_LIMIT_PROP_VEC3(maxComputeWorkGroupSize);
  ss << "    }" << std::endl;
  ss << "  }" << std::endl;
  ;

  const VkPhysicalDeviceMemoryProperties& mem_props =
      physical_device_.memory_properties;

  ss << "  Memory Info {" << std::endl;
  ss << "    Memory Types [" << std::endl;
  for (const auto i : c10::irange(mem_props.memoryTypeCount)) {
    ss << "      "
       << " [Heap " << mem_props.memoryTypes[i].heapIndex << "] "
       << get_memory_properties_str(mem_props.memoryTypes[i].propertyFlags)
       << std::endl;
  }
  ss << "    ]" << std::endl;
  ss << "    Memory Heaps [" << std::endl;
  for (const auto i : c10::irange(mem_props.memoryHeapCount)) {
    ss << "      " << mem_props.memoryHeaps[i].size << std::endl;
  }
  ss << "    ]" << std::endl;
  ss << "  }" << std::endl;

  ss << "  Queue Families {" << std::endl;
  for (const VkQueueFamilyProperties& queue_family_props :
       physical_device_.queue_families) {
    ss << "    (" << queue_family_props.queueCount << " Queues) "
       << get_queue_family_properties_str(queue_family_props.queueFlags)
       << std::endl;
  }
  ss << "  }" << std::endl;
  ss << "  VkDevice: " << device_.handle_ << std::endl;
  ss << "  Compute Queues [" << std::endl;
  for (const Adapter::Queue& compute_queue : queues_) {
    ss << "    Family " << compute_queue.family_index << ", Queue "
       << compute_queue.queue_index << ": " << compute_queue.handle
       << std::endl;
    ;
  }
  ss << "  ]" << std::endl;
  ss << "}";

  return ss.str();
}

std::ostream& operator<<(std::ostream& os, const Adapter& adapter) {
  os << adapter.stringize() << std::endl;
  return os;
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
