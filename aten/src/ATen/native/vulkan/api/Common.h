#pragma once

#include <ATen/ATen.h>

#ifdef USE_VULKAN_WRAPPER
#include <vulkan_wrapper.h>
#else
#include <vulkan/vulkan.h>
#endif

#define VK_CHECK(function)                                               \
  {                                                                      \
    const VkResult result = (function);                                  \
    TORCH_CHECK(result == VK_SUCCESS, "Vulkan error VkResult:", result); \
  }

#define VK_DELETER(Handle) at::native::vulkan::detail::destroy_##Handle
#define VK_DELETER_DECLARE_DISPATCHABLE(Handle) void destroy_##Handle(const Vk##Handle handle)
#define VK_DELETER_DECLARE_NON_DISPATCHABLE(Handle)   \
  class destroy_##Handle final {                      \
   public:                                            \
    explicit destroy_##Handle(const VkDevice device); \
    void operator()(const Vk##Handle handle) const;   \
   private:                                           \
    VkDevice device_;                                 \
  };

namespace at {
namespace native {
namespace vulkan {
namespace detail {

VK_DELETER_DECLARE_DISPATCHABLE(Instance);
VK_DELETER_DECLARE_DISPATCHABLE(Device);
VK_DELETER_DECLARE_NON_DISPATCHABLE(Semaphore);
VK_DELETER_DECLARE_NON_DISPATCHABLE(Fence);
VK_DELETER_DECLARE_NON_DISPATCHABLE(Buffer);
VK_DELETER_DECLARE_NON_DISPATCHABLE(Image);
VK_DELETER_DECLARE_NON_DISPATCHABLE(Event);
VK_DELETER_DECLARE_NON_DISPATCHABLE(BufferView);
VK_DELETER_DECLARE_NON_DISPATCHABLE(ImageView);
VK_DELETER_DECLARE_NON_DISPATCHABLE(ShaderModule);
VK_DELETER_DECLARE_NON_DISPATCHABLE(PipelineCache);
VK_DELETER_DECLARE_NON_DISPATCHABLE(PipelineLayout);
VK_DELETER_DECLARE_NON_DISPATCHABLE(Pipeline);
VK_DELETER_DECLARE_NON_DISPATCHABLE(DescriptorSetLayout);
VK_DELETER_DECLARE_NON_DISPATCHABLE(Sampler);
VK_DELETER_DECLARE_NON_DISPATCHABLE(DescriptorPool);
VK_DELETER_DECLARE_NON_DISPATCHABLE(CommandPool);

template <typename Type, typename Deleter>
using Handle = std::unique_ptr<typename std::remove_pointer<Type>::type, Deleter>;

} // namespace detail
} // namespace vulkan
} // namespace native
} // namespace at
