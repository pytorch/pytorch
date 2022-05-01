#include <ATen/native/vulkan/api/Adapter.h>
#include <iomanip>
#include <sstream>

namespace at {
namespace native {
namespace vulkan {
namespace api {

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

//
// Print utils
//

std::string get_device_type_str(const VkPhysicalDeviceType type) {
  switch(type) {
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
      return "INTEGRATED_GPU";
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
      return "DISCRETE_GPU";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
      return "VIRTUAL_GPU";
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
      return "CPU";
    default:
      return "UNKOWN";
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

Adapter::Adapter(const VkPhysicalDevice handle, const uint32_t num_queues)
  : physical_handle_(handle),
    properties_{},
    memory_properties_{},
    queue_families_{},
    num_requested_queues_{num_queues},
    queue_usage_{},
    handle_(VK_NULL_HANDLE),
    queues_{},
    num_compute_queues_{},
    has_unified_memory_{false},
    timestamp_compute_and_graphics_{false},
    timestamp_period_{0.f} {
  // This should never happen, but double check to be safe
  TORCH_CHECK(
      VK_NULL_HANDLE != physical_handle_,
      "Pytorch Vulkan Adapter: VK_NULL_HANDLE passed to Adapter constructor!")

  vkGetPhysicalDeviceProperties(physical_handle_, &properties_);
  vkGetPhysicalDeviceMemoryProperties(physical_handle_, &memory_properties_);

  timestamp_compute_and_graphics_ = properties_.limits.timestampComputeAndGraphics;
  timestamp_period_ = properties_.limits.timestampPeriod;

  // Check if there are any memory types have both the HOST_VISIBLE and the
  // DEVICE_LOCAL property flags
  const VkMemoryPropertyFlags unified_memory_flags =
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
  for (const uint32_t i : c10::irange(memory_properties_.memoryTypeCount)) {
    if (memory_properties_.memoryTypes[i].propertyFlags | unified_memory_flags) {
      has_unified_memory_ = true;
      break;
    }
  }

  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(
      physical_handle_, &queue_family_count, nullptr);

  queue_families_.resize(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(
      physical_handle_, &queue_family_count, queue_families_.data());

  // Find the total number of compute queues
  for (const uint32_t family_i : c10::irange(queue_families_.size())) {
    const VkQueueFamilyProperties& properties = queue_families_[family_i];
    // Check if this family has compute capability
    if (properties.queueFlags & VK_QUEUE_COMPUTE_BIT) {
      num_compute_queues_ += properties.queueCount;
    }
  }

  queue_usage_.reserve(num_requested_queues_);
  queues_.reserve(num_requested_queues_);
}

Adapter::Adapter(Adapter&& other) noexcept
  : physical_handle_(other.physical_handle_),
    properties_(other.properties_),
    memory_properties_(other.memory_properties_),
    queue_families_(std::move(other.queue_families_)),
    num_requested_queues_(other.num_requested_queues_),
    queue_usage_(std::move(other.queue_usage_)),
    handle_(other.handle_),
    queues_(std::move(other.queues_)),
    num_compute_queues_(other.num_compute_queues_),
    has_unified_memory_(other.has_unified_memory_),
    timestamp_compute_and_graphics_(other.timestamp_compute_and_graphics_),
    timestamp_period_(other.timestamp_period_) {
  other.physical_handle_ = VK_NULL_HANDLE;
  other.handle_ = VK_NULL_HANDLE;
}

Adapter::~Adapter() {
  if C10_LIKELY(VK_NULL_HANDLE == handle_) {
    return;
  }
  vkDestroyDevice(handle_, nullptr);
  handle_ = VK_NULL_HANDLE;
}

void Adapter::init_device() {
  // It is possible that multiple threads will attempt to initialize the device
  // simultaneously, so lock the mutex before initializing
  std::lock_guard<std::mutex> lock(mutex_);

  // Do not initialize the device if there are no compute queues available
  TORCH_CHECK(
      num_compute_queues_ > 0,
      "Pytorch Vulkan Adapter: Cannot initialize Adapter as this device does not "
      "have any queues that support compute!")

  // This device has already been initialized, no-op
  if C10_LIKELY(VK_NULL_HANDLE != handle_) {
    return;
  }

  //
  // Find compute queues up to the requested number of queues
  //

  std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
  queue_create_infos.reserve(num_requested_queues_);

  std::vector<std::pair<uint32_t, uint32_t>> queues_to_get;
  queues_to_get.reserve(num_requested_queues_);

  uint32_t remaining_queues = num_requested_queues_;
  for (const uint32_t family_i : c10::irange(queue_families_.size())) {
    const VkQueueFamilyProperties& properties = queue_families_[family_i];
    // Check if this family has compute capability
    if (properties.queueFlags & VK_QUEUE_COMPUTE_BIT) {
      const uint32_t queues_to_init = std::min(
          remaining_queues, properties.queueCount);

      const std::vector<float> queue_priorities(queues_to_init, 1.0f);
      queue_create_infos.push_back({
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,  // sType
        nullptr,  // pNext
        0u,  // flags
        family_i,  // queueFamilyIndex
        queues_to_init,  // queueCount
        queue_priorities.data(),  // pQueuePriorities
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

  //
  // Create the VkDevice
  //

  std::vector<const char*> requested_device_extensions {
  #ifdef VK_KHR_portability_subset
    VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME,
  #endif
  };

  std::vector<const char*> enabled_device_extensions;
  find_requested_device_extensions(
      physical_handle_, enabled_device_extensions, requested_device_extensions);

  const VkDeviceCreateInfo device_create_info{
    VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,  // sType
    nullptr,  // pNext
    0u,  // flags
    static_cast<uint32_t>(queue_create_infos.size()),  // queueCreateInfoCount
    queue_create_infos.data(),  // pQueueCreateInfos
    0u,  // enabledLayerCount
    nullptr,  // ppEnabledLayerNames
    static_cast<uint32_t>(enabled_device_extensions.size()),  // enabledExtensionCount
    enabled_device_extensions.data(),  // ppEnabledExtensionNames
    nullptr,  // pEnabledFeatures
  };

  const VkResult device_create_res = vkCreateDevice(
      physical_handle_, &device_create_info, nullptr, &handle_);
  // If device was not created successfully, ensure handle_ is invalid and throw
  if (VK_SUCCESS != device_create_res) {
    handle_ = VK_NULL_HANDLE;
    VK_CHECK(device_create_res);
  }

#ifdef USE_VULKAN_VOLK
  volkLoadDevice(handle_);
#endif

  //
  // Obtain handles for the created queues and initialize queue usage heuristic
  //

  for (const std::pair<uint32_t, uint32_t>& queue_idx : queues_to_get) {
    VkQueue queue_handle = VK_NULL_HANDLE;
    VkQueueFlags flags = queue_families_[queue_idx.first].queueFlags;
    vkGetDeviceQueue(
        handle_, queue_idx.first, queue_idx.second, &queue_handle);
    queues_.push_back({queue_idx.first, queue_idx.second, flags, queue_handle});
    // Initial usage value
    queue_usage_.push_back(0);
  }
}

Adapter::Queue Adapter::request_queue() {
  // Lock the mutex as multiple threads can request a queue at the same time
  std::lock_guard<std::mutex> lock(mutex_);

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
      std::lock_guard<std::mutex> lock(mutex_);
      queue_usage_[i] -= 1;
      break;
    }
  }
}

std::string Adapter::stringize() const {
  std::stringstream ss;

  uint32_t v_major = VK_VERSION_MAJOR(properties_.apiVersion);
  uint32_t v_minor = VK_VERSION_MINOR(properties_.apiVersion);
  std::string device_type = get_device_type_str(properties_.deviceType);
  VkPhysicalDeviceLimits limits = properties_.limits;

  ss << "{" << std::endl;
  ss << "  Physical Device Info {" << std::endl;
  ss << "    apiVersion:    " << v_major << "." << v_minor << std::endl;
  ss << "    driverversion: " << properties_.driverVersion << std::endl;
  ss << "    deviceType:    " << device_type << std::endl;
  ss << "    deviceName:    " << properties_.deviceName << std::endl;

#define PRINT_LIMIT_PROP(name) \
  ss << "      " << std::left << std::setw(36) << #name << limits.name << std::endl;

#define PRINT_LIMIT_PROP_VEC3(name) \
  ss << "      " << std::left << std::setw(36) << #name \
  << limits.name[0] << "," \
  << limits.name[1] << "," \
  << limits.name[2] << std::endl;

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
  ss << "  }" << std::endl;;

  const VkPhysicalDeviceMemoryProperties& mem_props = memory_properties_;
  ss << "  Memory Info {" << std::endl;
  ss << "    Memory Types [" << std::endl;
  for (int i = 0; i < mem_props.memoryTypeCount; ++i) {
  ss << "      " << " [Heap " << mem_props.memoryTypes[i].heapIndex << "] "
               << get_memory_properties_str(mem_props.memoryTypes[i].propertyFlags)
               << std::endl;
  }
  ss << "    ]" << std::endl;
  ss << "    Memory Heaps [" << std::endl;
  for (int i = 0; i < mem_props.memoryHeapCount; ++i) {
  ss << "      " << mem_props.memoryHeaps[i].size << std::endl;
  }
  ss << "    ]" << std::endl;
  ss << "  }" << std::endl;

  ss << "  Queue Families {" << std::endl;
  for (const VkQueueFamilyProperties& queue_family_props : queue_families_) {
  ss << "    (" << queue_family_props.queueCount << " Queues) "
     << get_queue_family_properties_str(queue_family_props.queueFlags) << std::endl;
  }
  ss << "  }" << std::endl;
  ss << "  VkDevice: " << handle_ << std::endl;
  ss << "  Compute Queues [" << std::endl;
  for (const Adapter::Queue& compute_queue : queues_) {
  ss << "    Family " << compute_queue.family_index
     << ", Queue " << compute_queue.queue_index
     << ": " << compute_queue.handle << std::endl;;
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
