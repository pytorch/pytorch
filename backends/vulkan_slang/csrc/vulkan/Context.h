#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace vulkan {

struct DeviceCapabilities {
    bool float16 = false;
    bool int8 = false;
    bool float64 = false;
    uint32_t subgroup_size = 0;
    bool cooperative_matrix = false;
    uint32_t max_workgroup_size = 0;
    uint32_t max_compute_shared_memory = 0;
    std::string device_name;
    uint32_t device_type = 0; // VkPhysicalDeviceType
};

class Context {
public:
    static Context& instance();

    // No copy/move
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    // Device management
    uint32_t device_count() const;
    void set_device(uint32_t index);
    uint32_t current_device() const;

    // Vulkan handles for current device
    VkInstance vk_instance() const { return instance_; }
    VkPhysicalDevice physical_device(uint32_t index = UINT32_MAX) const;
    VkDevice device(uint32_t index = UINT32_MAX) const;
    VkQueue compute_queue(uint32_t index = UINT32_MAX) const;
    uint32_t compute_queue_family(uint32_t index = UINT32_MAX) const;
    VmaAllocator allocator(uint32_t index = UINT32_MAX) const;

    const DeviceCapabilities& capabilities(uint32_t index = UINT32_MAX) const;
    std::string device_name(uint32_t index = UINT32_MAX) const;

    bool is_available() const { return !devices_.empty(); }

    // Release all Vulkan resources held by other singletons before
    // destroying VkDevice. Called automatically from the destructor.
    void shutdown();

private:
    Context();
    ~Context();

    void init_instance();
    void init_devices();
    void init_device(uint32_t index);

    static VkBool32 VKAPI_CALL debug_callback(
        VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT type,
        const VkDebugUtilsMessengerCallbackDataEXT* data,
        void* user_data);

    bool shutdown_done_ = false;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;

    struct DeviceState {
        VkPhysicalDevice physical = VK_NULL_HANDLE;
        VkDevice logical = VK_NULL_HANDLE;
        VkQueue compute_queue = VK_NULL_HANDLE;
        uint32_t compute_queue_family = 0;
        VmaAllocator allocator = VK_NULL_HANDLE;
        DeviceCapabilities caps;
    };

    std::vector<DeviceState> devices_;
    uint32_t current_device_ = 0;
    mutable std::mutex mutex_;
};

} // namespace vulkan
