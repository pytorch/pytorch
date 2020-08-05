#include <ATen/native/vulkan/api/Descriptor.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

Descriptor::Cache::Cache(const VkDevice device, const VkDescriptorPool descriptor_pool)
  : device_(device),
    descriptor_pool_(descriptor_pool) {
}

VkDescriptorSet Descriptor::Cache::allocate(
    const VkDescriptorSetLayout descriptor_set_layout) {
  const VkDescriptorSetAllocateInfo descriptor_set_allocate_info{
    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    nullptr,
    descriptor_pool_,
    1u,
    &descriptor_set_layout,
  };

  VkDescriptorSet descriptor_set{};
  VK_CHECK(vkAllocateDescriptorSets(
      device_, &descriptor_set_allocate_info, &descriptor_set));

  return descriptor_set;
}

void Descriptor::Cache::purge() {
  VK_CHECK(vkResetDescriptorPool(device_, descriptor_pool_, 0u));
}

Descriptor::Pool::Factory::Factory(const VkDevice device)
  : device_(device) {
}

typename Descriptor::Pool::Factory::Handle Descriptor::Pool::Factory::operator()(
  const Descriptor& descriptor) const {
  static_assert(
      sizeof(Descriptor::Size) == sizeof(VkDescriptorPoolSize),
      "This implementation assumes a Descriptor::Size's memory layout is the same"
      "as VkDescriptorPoolSize.  A copy needs to be performed otherwise.");

  const VkDescriptorPoolCreateInfo descriptor_pool_create_info{
    VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    nullptr,
    0u, /* Do not use VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT */
    descriptor.capacity,
    descriptor.sizes.size(),
    reinterpret_cast<const VkDescriptorPoolSize*>(descriptor.sizes.data()),
  };

  VkDescriptorPool descriptor_pool{};
  VK_CHECK(vkCreateDescriptorPool(
      device_, &descriptor_pool_create_info, nullptr, &descriptor_pool));

  return Handle{
    descriptor_pool,
    Deleter(device_),
  };
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
