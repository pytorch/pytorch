#include <ATen/native/vulkan/api/Descriptor.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {
namespace {

VkDescriptorPool create_descriptor_pool(const VkDevice device) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device,
      "Invalid Vulkan device!");

  const struct {
    uint32_t capacity;
    c10::SmallVector<VkDescriptorPoolSize, 4u> sizes;
  } descriptor {
    1024u,
    {
      /*
        Buffers
      */

      {
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        1024u,
      },
      {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        1024u,
      },

      /*
        Images
      */

      {
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        1024u,
      },
      {
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        1024u,
      },
    },
  };

  const VkDescriptorPoolCreateInfo descriptor_pool_create_info{
    VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    nullptr,
    0u,
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

void allocate_descriptor_sets(
    const VkDevice device,
    const VkDescriptorPool descriptor_pool,
    const VkDescriptorSetLayout descriptor_set_layout,
    VkDescriptorSet* const descriptor_sets,
    const uint32_t count) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device,
      "Invalid Vulkan device!");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      descriptor_pool,
      "Invalid Vulkan descriptor pool!");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      descriptor_set_layout,
      "Invalid Vulkan descriptor set layout!");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      descriptor_sets && (count > 0u),
      "Invalid usage!");

  const std::vector<VkDescriptorSetLayout> descriptor_set_layouts{
    count,
    descriptor_set_layout,
  };

  const VkDescriptorSetAllocateInfo descriptor_set_allocate_info{
    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    nullptr,
    descriptor_pool,
    descriptor_set_layouts.size(),
    descriptor_set_layouts.data(),
  };

  VK_CHECK(vkAllocateDescriptorSets(
      device,
      &descriptor_set_allocate_info,
      descriptor_sets));
}

} // namespace

Descriptor::Set::Set(
    const VkDevice device,
    VkDescriptorSet descriptor_set,
    const Shader::Layout::Signature& shader_layout_signature)
  : device_(device),
    descriptor_set_(descriptor_set),
    shader_layout_signature_(shader_layout_signature),
    bindings_{} {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device_,
      "Invalid Vulkan device!");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      descriptor_set_,
      "Invalid Vulkan descriptor set!");
}

Descriptor::Set::Set(Set&& set)
  : device_(std::move(set.device_)),
    descriptor_set_(std::move(set.descriptor_set_)),
    shader_layout_signature_(std::move(set.shader_layout_signature_)),
    bindings_(std::move(set.bindings_)) {
  set.invalidate();
}

Descriptor::Set& Descriptor::Set::operator=(Set&& set) {
  if (&set != this) {
    device_ = std::move(set.device_);
    descriptor_set_ = std::move(set.descriptor_set_);
    shader_layout_signature_ = std::move(set.shader_layout_signature_);
    bindings_ = std::move(set.bindings_);

    set.invalidate();
  };

  return *this;
}

Descriptor::Set& Descriptor::Set::bind(
    const uint32_t binding,
    const Resource::Buffer::Object& buffer) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device_ && descriptor_set_,
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
      device_ && descriptor_set_,
      "This descriptor set is in an invalid state! "
      "Potential reason: This descriptor set is moved from.");

  update({
      binding,
      shader_layout_signature_[binding],
      {
        .image = {
          image.sampler,
          image.view,
          [](const VkDescriptorType type, const VkImageLayout layout) {
            return (VK_DESCRIPTOR_TYPE_STORAGE_IMAGE == type) ?
                    VK_IMAGE_LAYOUT_GENERAL : layout;
          }(shader_layout_signature_[binding], image.layout),
        },
      },
    });

  return *this;
}

VkDescriptorSet Descriptor::Set::handle() const {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device_ && descriptor_set_,
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

    c10::SmallVector<VkWriteDescriptorSet, 6u> write_descriptor_sets;

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

    // Reset
    bindings_.dirty = false;
  }

  return descriptor_set_;
}

void Descriptor::Set::invalidate() {
  device_ = VK_NULL_HANDLE;
  descriptor_set_ = VK_NULL_HANDLE;
}

void Descriptor::Set::update(const Item& item) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device_ && descriptor_set_,
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
    descriptor_pool_(std::move(pool.descriptor_pool_)),
    set_(std::move(pool.set_)) {
  pool.invalidate();
}

Descriptor::Pool& Descriptor::Pool::operator=(Pool&& pool) {
  if (&pool != this) {
    device_ = std::move(pool.device_);
    descriptor_pool_ = std::move(pool.descriptor_pool_);
    set_ = std::move(pool.set_);

    pool.invalidate();
  };

  return *this;
}

Descriptor::Pool::~Pool() {
  try {
    if (device_ && descriptor_pool_) {
      purge();
    }
  }
  catch (const std::exception& e) {
    TORCH_WARN(
        "Vulkan: Descriptor pool destructor raised an exception! Error: ",
        e.what());
  }
  catch (...) {
    TORCH_WARN(
        "Vulkan: Descriptor pool destructor raised an exception! "
        "Error: Unknown");
  }
}

Descriptor::Set Descriptor::Pool::allocate(
    const Shader::Layout::Object& shader_layout) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device_ && descriptor_pool_,
      "This descriptor pool is in an invalid state! "
      "Potential reason: This descriptor pool is moved from.");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      shader_layout,
      "Invalid Vulkan shader layout!");

  auto iterator = set_.layouts.find(shader_layout.handle);
  if (set_.layouts.cend() == iterator) {
    iterator = set_.layouts.insert({shader_layout.handle, {}}).first;
    iterator->second.pool.reserve(Configuration::kReserve);
  }

  auto& layout = iterator->second;

  if (layout.pool.size() == layout.in_use) {
    layout.pool.resize(
        layout.pool.size() +
        Configuration::kQuantum);

    allocate_descriptor_sets(
        device_,
        descriptor_pool_.get(),
        shader_layout.handle,
        layout.pool.data() + layout.in_use,
        Configuration::kQuantum);
  }

  return Set(
      device_,
      layout.pool[layout.in_use++],
      shader_layout.signature);
}

void Descriptor::Pool::purge() {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device_ && descriptor_pool_,
      "This descriptor pool is in an invalid state! "
      "Potential reason: This descriptor pool is moved from.");

  VK_CHECK(vkResetDescriptorPool(device_, descriptor_pool_.get(), 0u));
  set_.layouts.clear();
}

void Descriptor::Pool::invalidate() {
  device_ = VK_NULL_HANDLE;
  descriptor_pool_.reset();
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
