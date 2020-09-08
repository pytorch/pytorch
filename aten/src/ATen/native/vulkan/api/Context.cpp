#include <ATen/native/vulkan/api/Context.h>

#include <sstream>

namespace at {
namespace native {
namespace vulkan {
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

    constexpr const char* const requested_instance_layers[]{
        // "VK_LAYER_LUNARG_api_dump",
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

    constexpr const char* const requested_instance_extensions[]{
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

  constexpr VkApplicationInfo application_info{
    VK_STRUCTURE_TYPE_APPLICATION_INFO,
    nullptr,
    "PyTorch",
    0,
    "PyTorch",
    0,
    VK_API_VERSION_1_0,
  };

  const VkInstanceCreateInfo instance_create_info{
    VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    nullptr,
    0u,
    &application_info,
    static_cast<uint32_t>(enabled_instance_layers.size()),
    enabled_instance_layers.data(),
    static_cast<uint32_t>(enabled_instance_extensions.size()),
    enabled_instance_extensions.data(),
  };

  VkInstance instance{};
  VK_CHECK(vkCreateInstance(&instance_create_info, nullptr, &instance));

  return instance;
}

VkDebugReportCallbackEXT create_debug_report_callback(
    const VkInstance instance,
    const bool enable_validation_layers) {
  if (!enable_validation_layers) {
    return VkDebugReportCallbackEXT{};
  }

  const VkDebugReportCallbackCreateInfoEXT debugReportCallbackCreateInfo{
    VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT,
    nullptr,
    VK_DEBUG_REPORT_INFORMATION_BIT_EXT |
      VK_DEBUG_REPORT_WARNING_BIT_EXT |
      VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT |
      VK_DEBUG_REPORT_ERROR_BIT_EXT |
      VK_DEBUG_REPORT_DEBUG_BIT_EXT,
    debug_report_callback_fn,
    nullptr,
  };

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

  return debug_report_callback;
}

VkPhysicalDevice acquire_physical_device(const VkInstance instance) {
  uint32_t device_count = 0;
  VK_CHECK(vkEnumeratePhysicalDevices(instance, &device_count, nullptr));
  TORCH_CHECK(device_count > 0, "Vulkan: Could not find a device with Vulkan support!");

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
      false,
      "Vulkan: Could not find a queue family that supports compute operations!");
}

VkDevice create_device(
    const VkPhysicalDevice physical_device,
    const uint32_t compute_queue_family_index) {
  const float queue_priorities = 1.0f;
  const VkDeviceQueueCreateInfo device_queue_create_info{
    VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    nullptr,
    0u,
    compute_queue_family_index,
    1u,
    &queue_priorities,
  };

  const VkDeviceCreateInfo device_create_info{
    VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    nullptr,
    0u,
    1u,
    &device_queue_create_info,
    0u,
    nullptr,
    0u,
    nullptr,
  };

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

} // namespace

Context::Context(const bool enable_validation_layers)
    : instance_(create_instance(enable_validation_layers), &VK_DELETER(Instance)),
      debug_report_callback_(
          create_debug_report_callback(instance(), enable_validation_layers),
          Debug(instance())),
      physical_device_(acquire_physical_device(instance())),
      physical_device_limits_(query_physical_device_physical_device_limits(physical_device())),
      compute_queue_family_index_(query_compute_queue_family_index(physical_device())),
      device_(create_device(physical_device(), compute_queue_family_index_), &VK_DELETER(Device)),
      queue_(acquire_queue(device(), compute_queue_family_index_)),
      command_(device(), {compute_queue_family_index_}),
      shader_(device()),
      pipeline_(device()),
      descriptor_(device()),
      resource_(instance(), physical_device(), device()) {
}

Context::Debug::Debug(const VkInstance instance)
  : instance_(instance) {
}

void Context::Debug::operator()(
    const VkDebugReportCallbackEXT debug_report_callback) const {
  if (debug_report_callback) {
    const auto vkDestroyDebugReportCallbackEXT =
      (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(
          instance_, "vkDestroyDebugReportCallbackEXT");

      TORCH_CHECK(
        vkDestroyDebugReportCallbackEXT,
        "Could not load vkDestroyDebugReportCallbackEXT");

      vkDestroyDebugReportCallbackEXT(
          instance_, debug_report_callback, nullptr);
  }
}

Context* initialize() {
  static const std::unique_ptr<Context> context([]() -> Context* {
#ifdef USE_VULKAN_WRAPPER
    if (!InitVulkan()) {
      TORCH_WARN("Vulkan: Wrapper Failed to InitVulkan");
      return nullptr;
    }
#endif

    try {
      return new Context(Configuration::kEnableValidationLayers);
    }
    catch (...) {
      return nullptr;
    }
  }());

  return context.get();
}

bool available() {
  return initialize();
}

Context* context() {
  Context* const context = initialize();
  TORCH_CHECK(context, "Vulkan: Backend not available on this platform!");

  return context;
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
