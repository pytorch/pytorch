#include <ATen/native/vulkan/api/Descriptor.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

const Descriptor::Pool::Descriptor Descriptor::Pool::kDefault{
  1024u,
  {
    // Note: It is OK for the sum of descriptors per type, below, to exceed
    // the max total figure above, but be concenious of memory consumption.
    // Considering how the descriptor pool must be frequently purged anyway
    // as a result of the impracticality of having enormous pools that
    // persist through the execution of the program, there is diminishing
    // return in increasing max counts.
    {
      /*
        Buffers
      */

      {
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        256u,
      },
      {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        256u,
      },

      /*
        Images
      */

      {
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        256u,
      },
      {
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        256u,
      },
    },
  },
};

Descriptor::Pool::Factory::Factory(const VkDevice device)
  : device_(device) {
}

typename Descriptor::Pool::Factory::Handle Descriptor::Pool::Factory::operator()(
  const Descriptor& descriptor) const {
  const VkDescriptorPoolCreateInfo descriptor_pool_create_info{
    VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    nullptr,
    0u, /* Do not use VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT */
    descriptor.capacity,
    static_cast<uint32_t>(descriptor.sizes.size()),
    descriptor.sizes.data(),
  };

  VkDescriptorPool descriptor_pool{};
  VK_CHECK(vkCreateDescriptorPool(
      device_, &descriptor_pool_create_info, nullptr, &descriptor_pool));

  return Handle{
    descriptor_pool,
    Deleter(device_),
  };
}

void Descriptor::Pool::purge(
    const VkDevice device,
    const VkDescriptorPool descriptor_pool) {
  VK_CHECK(vkResetDescriptorPool(device, descriptor_pool, 0u));
}

Descriptor::Factory::Factory(const VkDevice device, const VkDescriptorPool descriptor_pool)
  : device_(device),
    descriptor_pool_(descriptor_pool) {
}

VkDescriptorSet Descriptor::Factory::allocate(
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

void Descriptor::Factory::purge() {
  Pool::purge(device_, descriptor_pool_);
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
