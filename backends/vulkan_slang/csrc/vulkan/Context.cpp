#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>

#include "Context.h"
#include "Pipeline.h"
#include "../ops/dispatch.h"
#include "../backend/Allocator.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace vulkan {

// ── Debug callback ───────────────────────────────────────────────
VkBool32 VKAPI_CALL Context::debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT /*type*/,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void* /*user_data*/) {
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::cerr << "[Vulkan Validation] " << data->pMessage << std::endl;
    }
    return VK_FALSE;
}

// ── Singleton ────────────────────────────────────────────────────
Context& Context::instance() {
    static Context ctx;
    return ctx;
}

Context::Context() {
    init_instance();
    init_devices();
}

void Context::shutdown() {
    if (shutdown_done_) return;
    shutdown_done_ = true;

    // Wait for all GPU work to finish before destroying anything
    for (auto& d : devices_) {
        if (d.logical) {
            vkDeviceWaitIdle(d.logical);
        }
    }

    // Destroy dependent singletons in reverse-dependency order:
    // 1. Runtimes (Streams with fences, CommandPools, DescriptorPools)
    torch_vulkan::ops::cleanup_runtimes();
    // 2. Pipelines (VkPipeline, VkShaderModule, VkPipelineLayout, VkDescriptorSetLayout)
    PipelineCache::instance().clear();
    // 3. Allocator buffers (VkBuffer via VMA — must happen while VmaAllocator alive)
    torch_vulkan::VulkanAllocator::instance().release_all();
}

Context::~Context() {
    shutdown();

    for (auto& d : devices_) {
        if (d.allocator) {
            vmaDestroyAllocator(d.allocator);
            d.allocator = VK_NULL_HANDLE;
        }
        if (d.logical) {
            vkDestroyDevice(d.logical, nullptr);
            d.logical = VK_NULL_HANDLE;
        }
    }
    if (debug_messenger_) {
        auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
        if (func) func(instance_, debug_messenger_, nullptr);
        debug_messenger_ = VK_NULL_HANDLE;
    }
    if (instance_) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
}

// ── Instance creation ────────────────────────────────────────────
void Context::init_instance() {
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "torch_vulkan";
    app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.pEngineName = "torch_vulkan";
    app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.apiVersion = VK_API_VERSION_1_2;

    // Check for validation layer
    uint32_t layer_count = 0;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
    std::vector<VkLayerProperties> layers(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, layers.data());

    bool has_validation = false;
    const char* validation_layer = "VK_LAYER_KHRONOS_validation";
    for (const auto& layer : layers) {
        if (strcmp(layer.layerName, validation_layer) == 0) {
            has_validation = true;
            break;
        }
    }

    // Check for debug utils extension
    uint32_t ext_count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, nullptr);
    std::vector<VkExtensionProperties> exts(ext_count);
    vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, exts.data());

    bool has_debug_utils = false;
    for (const auto& ext : exts) {
        if (strcmp(ext.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0) {
            has_debug_utils = true;
            break;
        }
    }

    std::vector<const char*> enabled_layers;
    std::vector<const char*> enabled_extensions;

    if (has_validation) {
        enabled_layers.push_back(validation_layer);
    }
    if (has_debug_utils) {
        enabled_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;
    create_info.enabledLayerCount = static_cast<uint32_t>(enabled_layers.size());
    create_info.ppEnabledLayerNames = enabled_layers.data();
    create_info.enabledExtensionCount = static_cast<uint32_t>(enabled_extensions.size());
    create_info.ppEnabledExtensionNames = enabled_extensions.data();

    VkResult result = vkCreateInstance(&create_info, nullptr, &instance_);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan instance");
    }

    // Set up debug messenger
    if (has_validation && has_debug_utils) {
        VkDebugUtilsMessengerCreateInfoEXT dbg_info{};
        dbg_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        dbg_info.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        dbg_info.messageType =
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        dbg_info.pfnUserCallback = debug_callback;

        auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
        if (func) {
            func(instance_, &dbg_info, nullptr, &debug_messenger_);
        }
    }
}

// ── Device enumeration & creation ────────────────────────────────
void Context::init_devices() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    if (count == 0) return;

    std::vector<VkPhysicalDevice> physical_devices(count);
    vkEnumeratePhysicalDevices(instance_, &count, physical_devices.data());

    devices_.resize(count);
    for (uint32_t i = 0; i < count; i++) {
        devices_[i].physical = physical_devices[i];
        init_device(i);
    }
}

void Context::init_device(uint32_t index) {
    auto& dev = devices_[index];

    // Query properties
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(dev.physical, &props);
    dev.caps.device_name = props.deviceName;
    dev.caps.device_type = props.deviceType;
    dev.caps.max_workgroup_size = props.limits.maxComputeWorkGroupInvocations;
    dev.caps.max_compute_shared_memory = props.limits.maxComputeSharedMemorySize;

    // Query features
    VkPhysicalDeviceFeatures features;
    vkGetPhysicalDeviceFeatures(dev.physical, &features);
    dev.caps.float64 = features.shaderFloat64;

    // Check for float16 support
    VkPhysicalDeviceVulkan12Features vk12_features{};
    vk12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &vk12_features;
    vkGetPhysicalDeviceFeatures2(dev.physical, &features2);
    dev.caps.float16 = vk12_features.shaderFloat16;
    dev.caps.int8 = vk12_features.shaderInt8;

    // Subgroup properties
    VkPhysicalDeviceSubgroupProperties subgroup_props{};
    subgroup_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &subgroup_props;
    vkGetPhysicalDeviceProperties2(dev.physical, &props2);
    dev.caps.subgroup_size = subgroup_props.subgroupSize;

    // Find compute queue family
    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev.physical, &qf_count, nullptr);
    std::vector<VkQueueFamilyProperties> qf_props(qf_count);
    vkGetPhysicalDeviceQueueFamilyProperties(dev.physical, &qf_count, qf_props.data());

    // Prefer dedicated compute queue, fall back to any compute-capable
    uint32_t compute_family = UINT32_MAX;
    for (uint32_t i = 0; i < qf_count; i++) {
        if (qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            compute_family = i;
            // Prefer queue without graphics
            if (!(qf_props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                break;
            }
        }
    }
    if (compute_family == UINT32_MAX) {
        throw std::runtime_error("No compute queue family found on device " +
                                 dev.caps.device_name);
    }
    dev.compute_queue_family = compute_family;

    // Create logical device
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_ci{};
    queue_ci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_ci.queueFamilyIndex = compute_family;
    queue_ci.queueCount = 1;
    queue_ci.pQueuePriorities = &queue_priority;

    // Enable Vulkan 1.2 features we need
    VkPhysicalDeviceVulkan12Features enabled_vk12{};
    enabled_vk12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    enabled_vk12.shaderFloat16 = dev.caps.float16;
    enabled_vk12.shaderInt8 = dev.caps.int8;
    enabled_vk12.timelineSemaphore = VK_TRUE;
    enabled_vk12.bufferDeviceAddress = VK_FALSE;

    VkPhysicalDeviceFeatures enabled_features{};
    enabled_features.shaderFloat64 = dev.caps.float64;

    VkDeviceCreateInfo device_ci{};
    device_ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_ci.pNext = &enabled_vk12;
    device_ci.queueCreateInfoCount = 1;
    device_ci.pQueueCreateInfos = &queue_ci;
    device_ci.enabledExtensionCount = 0;
    device_ci.pEnabledFeatures = &enabled_features;

    VkResult result = vkCreateDevice(dev.physical, &device_ci, nullptr, &dev.logical);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create logical device for " +
                                 dev.caps.device_name);
    }

    vkGetDeviceQueue(dev.logical, compute_family, 0, &dev.compute_queue);

    // Create VMA allocator
    VmaVulkanFunctions vma_funcs{};
    vma_funcs.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vma_funcs.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo alloc_ci{};
    alloc_ci.vulkanApiVersion = VK_API_VERSION_1_2;
    alloc_ci.physicalDevice = dev.physical;
    alloc_ci.device = dev.logical;
    alloc_ci.instance = instance_;
    alloc_ci.pVulkanFunctions = &vma_funcs;

    result = vmaCreateAllocator(&alloc_ci, &dev.allocator);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create VMA allocator for " +
                                 dev.caps.device_name);
    }
}

// ── Public API ───────────────────────────────────────────────────
uint32_t Context::device_count() const {
    return static_cast<uint32_t>(devices_.size());
}

void Context::set_device(uint32_t index) {
    if (index >= devices_.size()) {
        throw std::runtime_error("Device index " + std::to_string(index) +
                                 " out of range (have " +
                                 std::to_string(devices_.size()) + ")");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    current_device_ = index;
}

uint32_t Context::current_device() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_device_;
}

VkPhysicalDevice Context::physical_device(uint32_t index) const {
    if (index == UINT32_MAX) index = current_device();
    return devices_.at(index).physical;
}

VkDevice Context::device(uint32_t index) const {
    if (index == UINT32_MAX) index = current_device();
    return devices_.at(index).logical;
}

VkQueue Context::compute_queue(uint32_t index) const {
    if (index == UINT32_MAX) index = current_device();
    return devices_.at(index).compute_queue;
}

uint32_t Context::compute_queue_family(uint32_t index) const {
    if (index == UINT32_MAX) index = current_device();
    return devices_.at(index).compute_queue_family;
}

VmaAllocator Context::allocator(uint32_t index) const {
    if (index == UINT32_MAX) index = current_device();
    return devices_.at(index).allocator;
}

const DeviceCapabilities& Context::capabilities(uint32_t index) const {
    if (index == UINT32_MAX) index = current_device();
    return devices_.at(index).caps;
}

std::string Context::device_name(uint32_t index) const {
    return capabilities(index).device_name;
}

} // namespace vulkan
