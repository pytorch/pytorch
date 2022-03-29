#include <ATen/native/vulkan/api/Adapter.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

namespace {

void find_requested_device_extensions(
    VkPhysicalDevice physical_device,
    std::vector<const char*>& enabled_extensions,
    std::vector<const char*>& requested_extensions) {
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

}

Adapter::Adapter(const VkPhysicalDevice handle)
  : physical_handle_(handle),
    properties_{},
    memory_properties_{},
    compute_queue_family_index_{},
    handle_(VK_NULL_HANDLE),
    queue_(VK_NULL_HANDLE) {
  vkGetPhysicalDeviceProperties(physical_handle_, &properties_);
  vkGetPhysicalDeviceMemoryProperties(physical_handle_, &memory_properties_);

  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(
      physical_handle_, &queue_family_count, nullptr);

  vkGetPhysicalDeviceQueueFamilyProperties(
      physical_handle_, &queue_family_count, queue_families_.data());

  // Find the compute family index
  for (const auto i : c10::irange(queue_families_.size())) {
    const VkQueueFamilyProperties& properties = queue_families_[i];
    // Selecting the first queue family with compute ability
    if (properties.queueCount > 0 && (properties.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
      compute_queue_family_index_ = i;
    }
  }
}

Adapter::Adapter(Adapter&& other) noexcept
  : physical_handle_(other.physical_handle_),
    properties_(other.properties_),
    memory_properties_(other.memory_properties_),
    queue_families_(std::move(other.queue_families_)),
    compute_queue_family_index_(other.compute_queue_family_index_),
    handle_(other.handle_),
    queue_(other.queue_) {
  other.physical_handle_ = VK_NULL_HANDLE;
  other.handle_ = VK_NULL_HANDLE;
  other.queue_ = VK_NULL_HANDLE;
}

Adapter::~Adapter() {
  if C10_LIKELY(VK_NULL_HANDLE == handle_) {
    return;
  }
  vkDestroyDevice(handle_, nullptr);
  handle_ = VK_NULL_HANDLE;
}

void Adapter::init_device() {
  if C10_LIKELY(VK_NULL_HANDLE == physical_handle_) {
    return;
  }
  // This device has already been initialized
  if C10_LIKELY(VK_NULL_HANDLE != handle_) {
    return;
  }

  const float queue_priorities = 1.0f;
  const VkDeviceQueueCreateInfo device_queue_create_info{
    VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,  // sType
    nullptr,  // pNext
    0u,  // flags
    compute_queue_family_index_,  // queueFamilyIndex
    1u,  // queueCount
    &queue_priorities,  // pQueuePriorities
  };

  std::vector<const char*> requested_device_extensions {
  #ifdef VK_KHR_portability_subset
    // https://vulkan.lunarg.com/doc/view/1.2.162.0/mac/1.2-extensions/vkspec.html#VUID-VkDeviceCreateInfo-pProperties-04451
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
    1u,  // queueCreateInfoCount
    &device_queue_create_info,  // pQueueCreateInfos
    0u,  // enabledLayerCount
    nullptr,  // ppEnabledLayerNames
    static_cast<uint32_t>(enabled_device_extensions.size()),  // enabledExtensionCount
    enabled_device_extensions.data(),  // ppEnabledExtensionNames
    nullptr,  // pEnabledFeatures
  };

  VK_CHECK(vkCreateDevice(physical_handle_, &device_create_info, nullptr, &handle_));
#ifdef USE_VULKAN_VOLK
  volkLoadDevice(handle_);
#endif

  vkGetDeviceQueue(handle_, compute_queue_family_index_, 0, &queue_);
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
