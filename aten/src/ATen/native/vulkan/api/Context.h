#pragma once

#include <ATen/native/vulkan/api/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace detail {
namespace api {

class VContext final {
 public:
  explicit VContext(bool enable_validation_layers);
  ~VContext() = default;
  VContext(const VContext&) = delete;
  VContext& operator=(const VContext&) = delete;
  VContext(VContext&&) = default;
  VContext& operator=(VContext&&) = default;

  inline VkInstance instance() const {
    return instance_.get();
  }

  inline VkPhysicalDevice physical_device() const {
    return physical_device_;
  }

  inline const VkPhysicalDeviceLimits& physical_device_limits() const {
    return physical_device_limits_;
  }

  inline VkDevice device() const {
    return device_.get();
  }

  inline VkQueue queue() const {
    return queue_;
  }

  inline VkCommandPool command_pool() const {
    return command_pool_.get();
  }

 private:
  Handle<VkInstance, decltype(&VK_DELETER(Instance))> instance_;
  VkPhysicalDevice physical_device_;
  VkPhysicalDeviceLimits physical_device_limits_;
  uint32_t compute_queue_family_index_;
  Handle<VkDevice, decltype(&VK_DELETER(Device))> device_;
  VkQueue queue_;
  Handle<VkCommandPool, VK_DELETER(CommandPool)> command_pool_;
};

bool available();
const VContext& context();

} // namespace api
} // namespace detail
} // namespace vulkan
} // namespace native
} // namespace at
