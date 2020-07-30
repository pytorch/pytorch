#include <ATen/native/vulkan/api/Context.h>

#include <sstream>

namespace at {
namespace native {
namespace vulkan {
namespace detail {
namespace api {
namespace {

struct Configuration final {
#ifndef DEBUG
  static constexpr bool kEnableValidationLayers = false;
#else
  static constexpr bool kEnableValidationLayers = true;
#endif
};

VKAPI_ATTR VkBool32 VKAPI_CALL debug_report_callback_fn(
    const VkDebugReportFlagsEXT flags,
    const VkDebugReportObjectTypeEXT /* object_type */,
    const uint64_t /* object */,
    const size_t /* location */,
    const int32_t message_code,
    const char* const layer_prefix,
    const char* const message,
    void* const /* user_data */) {
  std::stringstream stream;
  stream << layer_prefix << " " << message_code << " " << message << std::endl;
  const std::string log = stream.str();

  if (flags & VK_DEBUG_REPORT_ERROR_BIT_EXT) {
    LOG(ERROR) << log;
  } else if (flags & VK_DEBUG_REPORT_WARNING_BIT_EXT) {
    LOG(WARNING) << log;
  } else if (flags & VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT) {
    LOG(WARNING) << "Performance:" << log;
  } else if (flags & VK_DEBUG_REPORT_INFORMATION_BIT_EXT) {
    LOG(INFO) << log;
  } else if (flags & VK_DEBUG_REPORT_DEBUG_BIT_EXT) {
    LOG(INFO) << "Debug: " << log;
  }

  return VK_FALSE;
}

VkInstance create_instance(const bool enable_validation_layers) {
  std::vector<const char*> enabled_instance_layers;
  std::vector<const char*> enabled_instance_extensions;

  if (enable_validation_layers) {
    uint32_t instance_layers_count = 0;
    VK_CHECK(vkEnumerateInstanceLayerProperties(
        &instance_layers_count, nullptr));

    std::vector<VkLayerProperties> instance_layer_properties(
        instance_layers_count);

    VK_CHECK(vkEnumerateInstanceLayerProperties(
        &instance_layers_count,
        instance_layer_properties.data()));

    constexpr std::array<const char*, 6> requested_instance_layers{
        "VK_LAYER_GOOGLE_unique_objects",
        "VK_LAYER_GOOGLE_threading",
        "VK_LAYER_LUNARG_object_tracker",
        "VK_LAYER_LUNARG_core_validation",
        "VK_LAYER_LUNARG_parameter_validation",
        "VK_LAYER_KHRONOS_validation",
    };

    for (const auto& requested_instance_layer : requested_instance_layers) {
      for (const auto& layer : instance_layer_properties) {
        if (strcmp(requested_instance_layer, layer.layerName) == 0) {
          enabled_instance_layers.push_back(requested_instance_layer);
          break;
        }
      }
    }

    uint32_t instance_extension_count = 0;
    VK_CHECK(vkEnumerateInstanceExtensionProperties(
        nullptr, &instance_extension_count, nullptr));

    std::vector<VkExtensionProperties> instance_extension_properties(
        instance_extension_count);

    VK_CHECK(vkEnumerateInstanceExtensionProperties(
        nullptr, &instance_extension_count, instance_extension_properties.data()));

    constexpr std::array<const char*, 1> requested_instance_extensions{
      VK_EXT_DEBUG_REPORT_EXTENSION_NAME,
    };

    for (const auto& requested_instance_extension : requested_instance_extensions) {
      for (const auto& extension : instance_extension_properties) {
        if (strcmp(requested_instance_extension, extension.extensionName) == 0) {
          enabled_instance_extensions.push_back(requested_instance_extension);
          break;
        }
      }
    }
  }

  VkApplicationInfo applicationInfo{};
  applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  applicationInfo.pApplicationName = "PyTorch";
  applicationInfo.applicationVersion = 0;
  applicationInfo.pEngineName = "PyTorch";
  applicationInfo.engineVersion = 0;
  applicationInfo.apiVersion = VK_API_VERSION_1_0;

  VkInstanceCreateInfo instance_create_info{};
  instance_create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instance_create_info.flags = 0;
  instance_create_info.pApplicationInfo = &applicationInfo;
  instance_create_info.enabledLayerCount = enabled_instance_layers.size();
  instance_create_info.ppEnabledLayerNames = enabled_instance_layers.data();
  instance_create_info.enabledExtensionCount = enabled_instance_extensions.size();
  instance_create_info.ppEnabledExtensionNames = enabled_instance_extensions.data();

  VkInstance instance{};
  VK_CHECK(vkCreateInstance(&instance_create_info, nullptr, &instance));

  if (enable_validation_layers) {
    VkDebugReportCallbackCreateInfoEXT debugReportCallbackCreateInfo{};
    debugReportCallbackCreateInfo.sType =
        VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
    debugReportCallbackCreateInfo.flags =
        VK_DEBUG_REPORT_INFORMATION_BIT_EXT |
        VK_DEBUG_REPORT_WARNING_BIT_EXT |
        VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT |
        VK_DEBUG_REPORT_ERROR_BIT_EXT |
        VK_DEBUG_REPORT_DEBUG_BIT_EXT;
    debugReportCallbackCreateInfo.pfnCallback = &debug_report_callback_fn;

    const auto vkCreateDebugReportCallbackEXT =
        (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(
            instance, "vkCreateDebugReportCallbackEXT");

    TORCH_CHECK(
        vkCreateDebugReportCallbackEXT,
        "Could not load vkCreateDebugReportCallbackEXT");

    VkDebugReportCallbackEXT debug_report_callback{};
    VK_CHECK(vkCreateDebugReportCallbackEXT(
        instance,
        &debugReportCallbackCreateInfo,
        nullptr,
        &debug_report_callback));
  }

  return instance;
}

VkPhysicalDevice acquire_physical_device(const VkInstance instance) {
  uint32_t device_count = 0;
  VK_CHECK(vkEnumeratePhysicalDevices(instance, &device_count, nullptr));
  TORCH_CHECK(device_count > 0, "Vulkan: Could not find a device with vulkan support!");

  std::vector<VkPhysicalDevice> devices(device_count);
  VK_CHECK(vkEnumeratePhysicalDevices(instance, &device_count, devices.data()));

  return devices[0];
}

VkPhysicalDeviceLimits query_physical_device_physical_device_limits(
    const VkPhysicalDevice physical_device) {
  VkPhysicalDeviceProperties physical_device_properties{};
  vkGetPhysicalDeviceProperties(physical_device, &physical_device_properties);
  return physical_device_properties.limits;
}

uint32_t query_compute_queue_family_index(const VkPhysicalDevice physical_device) {
  uint32_t queue_family_count = 0;

  vkGetPhysicalDeviceQueueFamilyProperties(
      physical_device, &queue_family_count, nullptr);

  TORCH_CHECK(
      queue_family_count > 0, "Vulkan: Invalid number of queue families!");

  std::vector<VkQueueFamilyProperties> queue_families_properties(
    queue_family_count);

  vkGetPhysicalDeviceQueueFamilyProperties(
      physical_device, &queue_family_count, queue_families_properties.data());

  for (uint32_t i = 0; i < queue_families_properties.size(); ++i) {
    const VkQueueFamilyProperties& properties = queue_families_properties[i];
    if (properties.queueCount > 0 && (properties.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
      return i;
    }
  }

  TORCH_CHECK(
      false, "Vulkan: Could not find a queue family that supports operations!");
}

VkDevice create_device(
    const VkPhysicalDevice physical_device,
    const uint32_t compute_queue_family_index) {
  VkDeviceQueueCreateInfo queue_create_info{};
  queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_create_info.queueFamilyIndex = compute_queue_family_index;
  queue_create_info.queueCount = 1;
  const float queue_properties = 1.0f;
  queue_create_info.pQueuePriorities = &queue_properties;

  VkDeviceCreateInfo device_create_info{};
  device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_create_info.queueCreateInfoCount = 1;
  device_create_info.pQueueCreateInfos = &queue_create_info;

  VkDevice device{};
  VK_CHECK(vkCreateDevice(physical_device, &device_create_info, nullptr, &device));

  return device;
}

VkQueue acquire_queue(
    const VkDevice device,
    const uint32_t compute_queue_family_index) {
  VkQueue queue{};
  vkGetDeviceQueue(device, compute_queue_family_index, 0, &queue);
  return queue;
}

VkCommandPool create_command_pool(
    const VkDevice device,
    const uint32_t compute_queue_family_index) {
  VkCommandPoolCreateInfo command_pool_create_info{};
  command_pool_create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  command_pool_create_info.flags = 0;
  command_pool_create_info.queueFamilyIndex = compute_queue_family_index;

  VkCommandPool command_pool{};
  VK_CHECK(vkCreateCommandPool(device, &command_pool_create_info, nullptr, &command_pool));

  return command_pool;
}

} // namespace

VContext::VContext(const bool enable_validation_layers)
    : instance_(create_instance(enable_validation_layers), &VK_DELETER(Instance)),
      physical_device_(acquire_physical_device(instance())),
      physical_device_limits_(query_physical_device_physical_device_limits(physical_device())),
      compute_queue_family_index_(query_compute_queue_family_index(physical_device())),
      device_(create_device(physical_device(), compute_queue_family_index_), &VK_DELETER(Device)),
      queue_(acquire_queue(device(), compute_queue_family_index_)),
      command_pool_(create_command_pool(device(), compute_queue_family_index_), VK_DELETER(CommandPool)(device())) {
}

const VContext* initialize() {
  static const std::unique_ptr<VContext> context([]() -> VContext* {
#ifdef USE_VULKAN_WRAPPER
    if (!InitVulkan()) {
      TORCH_WARN("Vulkan Wrapper Failed to InitVulkan");
      return nullptr;
    }
#endif

    return new VContext(Configuration::kEnableValidationLayers);
  }());

  return context.get();
}

bool available() {
  return initialize();
}

const VContext& context() {
  return *initialize();
}

} // namespace api
} // namespace detail
} // namespace vulkan
} // namespace native
} // namespace at
