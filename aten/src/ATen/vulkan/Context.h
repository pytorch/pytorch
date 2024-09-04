#pragma once

#include <atomic>

#include <ATen/Tensor.h>

namespace at {
namespace vulkan {

struct VulkanImplInterface {
  virtual ~VulkanImplInterface() = default;
  virtual bool is_vulkan_available() const = 0;
  virtual at::Tensor& vulkan_copy_(at::Tensor& self, const at::Tensor& src)
      const = 0;
};

extern std::atomic<const VulkanImplInterface*> g_vulkan_impl_registry;

class VulkanImplRegistrar {
 public:
  explicit VulkanImplRegistrar(VulkanImplInterface*);
};

at::Tensor& vulkan_copy_(at::Tensor& self, const at::Tensor& src);
} // namespace vulkan

namespace native {
  bool is_vulkan_available();
}// namespace native

} // namespace at
