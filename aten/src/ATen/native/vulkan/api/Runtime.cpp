#include <ATen/native/vulkan/api/Adapter.h>
#include <ATen/native/vulkan/api/Runtime.h>
#include <c10/util/Logging.h>
#include <c10/util/irange.h>

#include <sstream>

namespace at {
namespace native {
namespace vulkan {
namespace api {

namespace {

void find_requested_layers_and_extensions(
    std::vector<const char*>& enabled_layers,
    std::vector<const char*>& enabled_extensions,
    const std::vector<const char*>& requested_layers,
    const std::vector<const char*>& requested_extensions) {
  // Get supported instance layers
  uint32_t layer_count = 0;
  VK_CHECK(vkEnumerateInstanceLayerProperties(&layer_count, nullptr));

  std::vector<VkLayerProperties> layer_properties(layer_count);
  VK_CHECK(vkEnumerateInstanceLayerProperties(
      &layer_count, layer_properties.data()));

  // Search for requested layers
  for (const auto& requested_layer : requested_layers) {
    for (const auto& layer : layer_properties) {
      if (strcmp(requested_layer, layer.layerName) == 0) {
        enabled_layers.push_back(requested_layer);
        break;
      }
    }
  }

  // Get supported instance extensions
  uint32_t extension_count = 0;
  VK_CHECK(vkEnumerateInstanceExtensionProperties(
      nullptr, &extension_count, nullptr));

  std::vector<VkExtensionProperties> extension_properties(extension_count);
  VK_CHECK(vkEnumerateInstanceExtensionProperties(
      nullptr, &extension_count, extension_properties.data()));

  // Search for requested extensions
  for (const auto& requested_extension : requested_extensions) {
    for (const auto& extension : extension_properties) {
      if (strcmp(requested_extension, extension.extensionName) == 0) {
        enabled_extensions.push_back(requested_extension);
        break;
      }
    }
  }
}

VkInstance create_instance(const RuntimeConfiguration& config) {
  const VkApplicationInfo application_info{
      VK_STRUCTURE_TYPE_APPLICATION_INFO, // sType
      nullptr, // pNext
      "PyTorch Vulkan Backend", // pApplicationName
      0, // applicationVersion
      nullptr, // pEngineName
      0, // engineVersion
      VK_API_VERSION_1_0, // apiVersion
  };

  std::vector<const char*> enabled_layers;
  std::vector<const char*> enabled_extensions;

  if (config.enableValidationMessages) {
    std::vector<const char*> requested_layers{
        // "VK_LAYER_LUNARG_api_dump",
        "VK_LAYER_KHRONOS_validation",
    };
    std::vector<const char*> requested_extensions{
#ifdef VK_EXT_debug_report
        VK_EXT_DEBUG_REPORT_EXTENSION_NAME,
#endif /* VK_EXT_debug_report */
    };

    find_requested_layers_and_extensions(
        enabled_layers,
        enabled_extensions,
        requested_layers,
        requested_extensions);
  }

  const VkInstanceCreateInfo instance_create_info{
      VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      &application_info, // pApplicationInfo
      static_cast<uint32_t>(enabled_layers.size()), // enabledLayerCount
      enabled_layers.data(), // ppEnabledLayerNames
      static_cast<uint32_t>(enabled_extensions.size()), // enabledExtensionCount
      enabled_extensions.data(), // ppEnabledExtensionNames
  };

  VkInstance instance{};
  VK_CHECK(vkCreateInstance(&instance_create_info, nullptr, &instance));
  TORCH_CHECK(instance, "Invalid Vulkan instance!");

#ifdef USE_VULKAN_VOLK
  volkLoadInstance(instance);
#endif /* USE_VULKAN_VOLK */

  return instance;
}

std::vector<Runtime::DeviceMapping> create_physical_devices(
    const VkInstance instance) {
  if (VK_NULL_HANDLE == instance) {
    return std::vector<Runtime::DeviceMapping>();
  }

  uint32_t device_count = 0;
  VK_CHECK(vkEnumeratePhysicalDevices(instance, &device_count, nullptr));

  std::vector<VkPhysicalDevice> devices(device_count);
  VK_CHECK(vkEnumeratePhysicalDevices(instance, &device_count, devices.data()));

  std::vector<Runtime::DeviceMapping> device_mappings;
  device_mappings.reserve(device_count);
  for (const VkPhysicalDevice physical_device : devices) {
    device_mappings.emplace_back(PhysicalDevice(physical_device), -1);
  }

  return device_mappings;
}

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

VkDebugReportCallbackEXT create_debug_report_callback(
    const VkInstance instance,
    const RuntimeConfiguration config) {
  if (VK_NULL_HANDLE == instance || !config.enableValidationMessages) {
    return VkDebugReportCallbackEXT{};
  }

  const VkDebugReportCallbackCreateInfoEXT debugReportCallbackCreateInfo{
      VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT, // sType
      nullptr, // pNext
      VK_DEBUG_REPORT_INFORMATION_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT |
          VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT |
          VK_DEBUG_REPORT_ERROR_BIT_EXT |
          VK_DEBUG_REPORT_DEBUG_BIT_EXT, // flags
      debug_report_callback_fn, // pfnCallback
      nullptr, // pUserData
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

  TORCH_CHECK(debug_report_callback, "Invalid Vulkan debug report callback!");

  return debug_report_callback;
}

//
// Adapter selection methods
//

uint32_t select_first(const std::vector<Runtime::DeviceMapping>& devices) {
  if (devices.size() == 0) {
    TORCH_WARN(
        "Pytorch Vulkan Runtime: no device devices are available for selection!");
    return devices.size() + 1; // return out of range to signal invalidity
  }

  // Select the first adapter that has compute capability
  for (const uint32_t i : c10::irange(devices.size())) {
    if (devices[i].first.num_compute_queues > 0) {
      return i;
    }
  }

  TORCH_WARN("Pytorch Vulkan Runtime: no device devices support compute!");
  return devices.size() + 1;
}

//
// Global runtime initialization
//

std::unique_ptr<Runtime> init_global_vulkan_runtime() {
  // Load Vulkan drivers
#if defined(USE_VULKAN_VOLK)
  if (VK_SUCCESS != volkInitialize()) {
    TORCH_WARN(
        "Pytorch Vulkan Runtime: Failed to load Vulkan driver using volkInitialize()! "
        "The global vulkan runtime is invalid.");
    return std::unique_ptr<Runtime>(nullptr);
  }
#elif defined(USE_VULKAN_WRAPPER)
  if (!InitVulkan()) {
    TORCH_WARN(
        "Pytorch Vulkan Runtime: Failed to load Vulkan driver using initVulkan()! "
        "The global vulkan runtime is invalid.");
    return std::unique_ptr<Runtime>(nullptr);
  }
#endif /* USE_VULKAN_VOLK, USE_VULKAN_WRAPPER */

  const bool enableValidationMessages =
#if defined(VULKAN_DEBUG)
      true;
#else
      false;
#endif /* VULKAN_DEBUG */
  const bool initDefaultDevice = true;
  const uint32_t numRequestedQueues = 1; // TODO: raise this value

  const RuntimeConfiguration default_config{
      enableValidationMessages,
      initDefaultDevice,
      AdapterSelector::First,
      numRequestedQueues,
  };

  try {
    return std::make_unique<Runtime>(Runtime(default_config));
  } catch (const c10::Error& e) {
    TORCH_WARN(
        "Pytorch Vulkan Runtime: Failed to initialize the global vulkan runtime! "
        "The global vulkan runtime is invalid. Error: ",
        e.what());
  } catch (const std::exception& e) {
    TORCH_WARN(
        "Pytorch Vulkan Runtime: Failed to initialize the global vulkan runtime! "
        "The global vulkan runtime is invalid. Error: ",
        e.what());
  } catch (...) {
    TORCH_WARN(
        "Pytorch Vulkan Runtime: Failed to initialize the global vulkan runtime! "
        "The global vulkan runtime is invalid. "
        "Error: Unknown");
  }

  return std::unique_ptr<Runtime>(nullptr);
}

} // namespace

Runtime::Runtime(const RuntimeConfiguration config)
    : config_(config),
      instance_(create_instance(config_)),
      device_mappings_(create_physical_devices(instance_)),
      adapters_{},
      default_adapter_i_(UINT32_MAX),
      debug_report_callback_(create_debug_report_callback(instance_, config_)) {
  // List of adapters will never exceed the number of physical devices
  adapters_.reserve(device_mappings_.size());

  if (config.initDefaultDevice) {
    try {
      switch (config.defaultSelector) {
        case AdapterSelector::First:
          default_adapter_i_ = create_adapter(select_first);
      }
    } catch (const c10::Error& e) {
      TORCH_WARN(
          "Pytorch Vulkan Runtime: Could not initialize default device! Error: ",
          e.what());
    } catch (const std::exception& e) {
      TORCH_WARN(
          "Pytorch Vulkan Runtime: Could not initialize default device! Error: ",
          e.what());
    } catch (...) {
      TORCH_WARN(
          "Pytorch Vulkan Runtime: Could not initialize default device! Error: "
          "Unknown.");
    }
  }
}

Runtime::~Runtime() {
  if C10_LIKELY (VK_NULL_HANDLE == instance_) {
    return;
  }

  // Clear adapters list to trigger device destruction before destroying
  // VkInstance
  adapters_.clear();

  // Instance must be destroyed last as its used to destroy the debug report
  // callback.
  if (debug_report_callback_) {
    const auto vkDestroyDebugReportCallbackEXT =
        (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(
            instance_, "vkDestroyDebugReportCallbackEXT");

    TORCH_CHECK(
        vkDestroyDebugReportCallbackEXT,
        "Pytorch Vulkan Runtime: Could not load vkDestroyDebugReportCallbackEXT "
        "when destroying debug_report_callback_");

    vkDestroyDebugReportCallbackEXT(instance_, debug_report_callback_, nullptr);

    debug_report_callback_ = {};
  }

  vkDestroyInstance(instance_, nullptr);
  instance_ = VK_NULL_HANDLE;
}

Runtime::Runtime(Runtime&& other) noexcept
    : config_(other.config_),
      instance_(other.instance_),
      adapters_(std::move(other.adapters_)),
      default_adapter_i_(other.default_adapter_i_),
      debug_report_callback_(other.debug_report_callback_) {
  other.instance_ = VK_NULL_HANDLE;
  other.debug_report_callback_ = {};
}

uint32_t Runtime::create_adapter(const Selector& selector) {
  TORCH_CHECK(
      device_mappings_.size() > 0,
      "Pytorch Vulkan Runtime: Could not initialize adapter because no "
      "devices were found by the Vulkan instance.");

  uint32_t physical_device_i = selector(device_mappings_);
  TORCH_CHECK(
      physical_device_i < device_mappings_.size(),
      "Pytorch Vulkan Runtime: no suitable device adapter was selected! "
      "Device could not be initialized");

  Runtime::DeviceMapping& device_mapping = device_mappings_[physical_device_i];
  // If an Adapter has already been created, return that
  int32_t adapter_i = device_mapping.second;
  if (adapter_i >= 0) {
    return adapter_i;
  }
  // Otherwise, create an adapter for the selected physical device
  adapter_i = utils::safe_downcast<uint32_t>(adapters_.size());
  adapters_.emplace_back(
      new Adapter(instance_, device_mapping.first, config_.numRequestedQueues));
  device_mapping.second = adapter_i;

  return adapter_i;
}

Runtime* runtime() {
  // The global vulkan runtime is declared as a static local variable within a
  // non-static function to ensure it has external linkage. If it were a global
  // static variable there would be one copy per translation unit that includes
  // Runtime.h as it would have internal linkage.
  static const std::unique_ptr<Runtime> p_runtime =
      init_global_vulkan_runtime();

  TORCH_CHECK(
      p_runtime,
      "Pytorch Vulkan Runtime: The global runtime could not be retrieved "
      "because it failed to initialize.");

  return p_runtime.get();
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
