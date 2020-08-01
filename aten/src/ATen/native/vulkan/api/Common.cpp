#include <ATen/native/vulkan/api/Common.h>

#define VK_DELETER_DEFINE_DISPATCHABLE(Handle)  \
  VK_DELETER_DECLARE_DISPATCHABLE(Handle) {     \
    if (C10_LIKELY(handle)) {                   \
      vkDestroy##Handle(handle, nullptr);       \
    }                                           \
  }

#define VK_DELETER_DEFINE_NON_DISPATCHABLE(Handle) \
  destroy_##Handle::destroy_##Handle(const VkDevice device)           \
    : device_(device) {                                               \
  }                                                                   \
                                                                      \
  void destroy_##Handle::operator()(const Vk##Handle handle) const {  \
    if (VK_NULL_HANDLE != handle) {                                   \
      vkDestroy##Handle(device_, handle, nullptr);                    \
    }                                                                 \
  }

namespace at {
namespace native {
namespace vulkan {
namespace api {

VK_DELETER_DEFINE_DISPATCHABLE(Instance);
VK_DELETER_DEFINE_DISPATCHABLE(Device);
VK_DELETER_DEFINE_NON_DISPATCHABLE(Semaphore);
VK_DELETER_DEFINE_NON_DISPATCHABLE(Fence);
VK_DELETER_DEFINE_NON_DISPATCHABLE(Buffer);
VK_DELETER_DEFINE_NON_DISPATCHABLE(Image);
VK_DELETER_DEFINE_NON_DISPATCHABLE(Event);
VK_DELETER_DEFINE_NON_DISPATCHABLE(BufferView);
VK_DELETER_DEFINE_NON_DISPATCHABLE(ImageView);
VK_DELETER_DEFINE_NON_DISPATCHABLE(ShaderModule);
VK_DELETER_DEFINE_NON_DISPATCHABLE(PipelineCache);
VK_DELETER_DEFINE_NON_DISPATCHABLE(PipelineLayout);
VK_DELETER_DEFINE_NON_DISPATCHABLE(Pipeline);
VK_DELETER_DEFINE_NON_DISPATCHABLE(DescriptorSetLayout);
VK_DELETER_DEFINE_NON_DISPATCHABLE(Sampler);
VK_DELETER_DEFINE_NON_DISPATCHABLE(DescriptorPool);
VK_DELETER_DEFINE_NON_DISPATCHABLE(CommandPool);

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
