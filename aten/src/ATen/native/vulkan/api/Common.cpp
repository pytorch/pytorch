#include <ATen/native/vulkan/api/Common.h>

#define VK_DELETER_DISPATCHABLE_DEFINE(Handle)  \
  VK_DELETER_DISPATCHABLE_DECLARE(Handle) {     \
    if (C10_LIKELY(VK_NULL_HANDLE != handle)) { \
      vkDestroy##Handle(handle, nullptr);       \
    }                                           \
  }

#define VK_DELETER_NON_DISPATCHABLE_DEFINE(Handle)                    \
  destroy_##Handle::destroy_##Handle(const VkDevice device)           \
    : device_(device) {                                               \
  }                                                                   \
                                                                      \
  void destroy_##Handle::operator()(const Vk##Handle handle) const {  \
    if (C10_LIKELY(VK_NULL_HANDLE != handle)) {                       \
      vkDestroy##Handle(device_, handle, nullptr);                    \
    }                                                                 \
  }

namespace at {
namespace native {
namespace vulkan {
namespace api {

VK_DELETER_DISPATCHABLE_DEFINE(Instance);
VK_DELETER_DISPATCHABLE_DEFINE(Device);
VK_DELETER_NON_DISPATCHABLE_DEFINE(Semaphore);
VK_DELETER_NON_DISPATCHABLE_DEFINE(Fence);
VK_DELETER_NON_DISPATCHABLE_DEFINE(Buffer);
VK_DELETER_NON_DISPATCHABLE_DEFINE(Image);
VK_DELETER_NON_DISPATCHABLE_DEFINE(Event);
VK_DELETER_NON_DISPATCHABLE_DEFINE(BufferView);
VK_DELETER_NON_DISPATCHABLE_DEFINE(ImageView);
VK_DELETER_NON_DISPATCHABLE_DEFINE(ShaderModule);
VK_DELETER_NON_DISPATCHABLE_DEFINE(PipelineCache);
VK_DELETER_NON_DISPATCHABLE_DEFINE(PipelineLayout);
VK_DELETER_NON_DISPATCHABLE_DEFINE(Pipeline);
VK_DELETER_NON_DISPATCHABLE_DEFINE(DescriptorSetLayout);
VK_DELETER_NON_DISPATCHABLE_DEFINE(Sampler);
VK_DELETER_NON_DISPATCHABLE_DEFINE(DescriptorPool);
VK_DELETER_NON_DISPATCHABLE_DEFINE(CommandPool);

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
