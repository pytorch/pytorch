#include <ATen/native/vulkan/api/Descriptor.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {
namespace {

VkDescriptorPool create_descriptor_pool(
    const VkDevice device) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device,
      "Invalid Vulkan device!");

  const struct {
    uint32_t capacity;
    c10::SmallVector<VkDescriptorPoolSize, 8u> sizes;
  } descriptor {
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
          768u,
        },
        {
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          768u,
        },

        /*
          Images
        */

        {
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          768u,
        },
        {
          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          768u,
        },
      },
    },
  };

  const VkDescriptorPoolCreateInfo descriptor_pool_create_info{
    VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    nullptr,
    0u, /* Do not use VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT. */
    descriptor.capacity,
    static_cast<uint32_t>(descriptor.sizes.size()),
    descriptor.sizes.data(),
  };

  VkDescriptorPool descriptor_pool{};
  VK_CHECK(vkCreateDescriptorPool(
      device,
      &descriptor_pool_create_info,
      nullptr,
      &descriptor_pool));

  TORCH_CHECK(
      descriptor_pool,
      "Invalid Vulkan descriptor pool!");

  return descriptor_pool;
}

VkDescriptorSet allocate_descriptor_set(
    const VkDevice device,
    const VkDescriptorPool descriptor_pool,
    const VkDescriptorSetLayout descriptor_set_layout) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device,
      "Invalid Vulkan device!");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      descriptor_pool,
      "Invalid Vulkan descriptor pool!");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      descriptor_set_layout,
      "Invalid Vulkan descriptor set layout!");

  const VkDescriptorSetAllocateInfo descriptor_set_allocate_info{
    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    nullptr,
    descriptor_pool,
    1u,
    &descriptor_set_layout,
  };

  VkDescriptorSet descriptor_set{};
  VK_CHECK(vkAllocateDescriptorSets(
      device,
      &descriptor_set_allocate_info,
      &descriptor_set));

  TORCH_CHECK(
      descriptor_set,
      "Invalid Vulkan descriptor set!");

  return descriptor_set;
}

} // namespace

Descriptor::Set::Set(
    const VkDevice device,
    const VkDescriptorPool descriptor_pool,
    const Shader::Layout::Object& shader_layout)
  : device_(device),
    descriptor_set_(
        allocate_descriptor_set(
            device_,
            descriptor_pool,
            shader_layout.handle)),
    shader_layout_signature_(shader_layout.signature),
    bindings_{} {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      descriptor_set_,
      "Invalid Vulkan descriptor set!");
}

void Descriptor::Set::update(const Item& item) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device_,
      "This descriptor set is in an invalid state! "
      "Potential reason: This descriptor set is moved from.");

  const auto items_itr = std::find_if(
      bindings_.items.begin(),
      bindings_.items.end(),
      [binding = item.binding](const Item& other) {
        return other.binding == binding;
      });

  if (bindings_.items.end() == items_itr) {
     bindings_.items.emplace_back(item);
  }
  else {
    *items_itr = item;
  }

  bindings_.dirty = true;
}

Descriptor::Set& Descriptor::Set::bind(
    const uint32_t binding,
    const Resource::Buffer::Object& buffer) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device_,
      "This descriptor set is in an invalid state! "
      "Potential reason: This descriptor set is moved from.");

  update({
      binding,
      shader_layout_signature_[binding],
      {
        .buffer = {
          buffer.handle,
          buffer.offset,
          buffer.range,
        },
      },
    });

  return *this;
}

Descriptor::Set& Descriptor::Set::bind(
    const uint32_t binding,
    const Resource::Image::Object& image) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device_,
      "This descriptor set is in an invalid state! "
      "Potential reason: This descriptor set is moved from.");

  update({
      binding,
      shader_layout_signature_[binding],
      {
        .image = {
          image.sampler,
          image.view,
          image.layout
        },
      },
    });

  return *this;
}

VkDescriptorSet Descriptor::Set::handle() const {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device_,
      "This descriptor set is in an invalid state! "
      "Potential reason: This descriptor set is moved from.");

  if (bindings_.dirty) {
    const auto is_buffer = [](const VkDescriptorType type) {
      switch (type) {
        case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
          return true;

        default:
          return false;
      }
    };

    const auto is_image = [](const VkDescriptorType type) {
      switch (type) {
        case VK_DESCRIPTOR_TYPE_SAMPLER:
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
        case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
          return true;

        default:
          return false;
      }
    };

    c10::SmallVector<VkWriteDescriptorSet, 8u> write_descriptor_sets;

    for (const Item& item : bindings_.items) {
      VkWriteDescriptorSet write{
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        nullptr,
        descriptor_set_,
        item.binding,
        0u,
        1u,
        item.type,
        nullptr,
        nullptr,
        nullptr,
      };

      if (is_buffer(item.type)) {
        write.pBufferInfo = &item.info.buffer;
      }
      else if (is_image(item.type)) {
        write.pImageInfo = &item.info.image;
      }

      write_descriptor_sets.emplace_back(write);
    }

    vkUpdateDescriptorSets(
        device_,
        write_descriptor_sets.size(),
        write_descriptor_sets.data(),
        0u,
        nullptr);

    bindings_.dirty = false;
  }

  return descriptor_set_;
}

Descriptor::Pool::Pool(const GPU& gpu)
  : device_(gpu.device),
    descriptor_pool_(
        create_descriptor_pool(gpu.device),
        VK_DELETER(DescriptorPool)(device_)) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device_,
      "Invalid Vulkan device!");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      descriptor_pool_,
      "Invalid Vulkan descriptor pool!");
}

Descriptor::Pool::Pool(Pool&& pool)
  : device_(std::move(pool.device_)),
    descriptor_pool_(std::move(pool.descriptor_pool_)) {
  pool.device_ = VK_NULL_HANDLE;
}

Descriptor::Pool& Descriptor::Pool::operator=(Pool&& pool) {
  if (&pool != this) {
    device_ = std::move(pool.device_);
    descriptor_pool_ = std::move(pool.descriptor_pool_);

    pool.device_ = VK_NULL_HANDLE;
  };

  return *this;
}

Descriptor::Set Descriptor::Pool::allocate(
    const Shader::Layout::Object& shader_layout)
{
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device_ && descriptor_pool_,
      "This descriptor pool is in an invalid state! "
      "Potential reason: This descriptor pool is moved from.");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      shader_layout,
      "Invalid Vulkan shader layout!");

  return Set(
      device_,
      descriptor_pool_.get(),
      shader_layout);
}

void Descriptor::Pool::purge() {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device_ && descriptor_pool_,
      "This descriptor pool is in an invalid state! "
      "Potential reason: This descriptor pool is moved from.");

  VK_CHECK(vkResetDescriptorPool(device_, descriptor_pool_.get(), 0u));
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
