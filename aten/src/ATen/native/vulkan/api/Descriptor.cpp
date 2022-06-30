#include <ATen/native/vulkan/api/Descriptor.h>
#include <ATen/native/vulkan/api/Utils.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

//
// DescriptorSet
//

DescriptorSet::DescriptorSet(
    const VkDevice device,
    const VkDescriptorSet handle,
    const ShaderLayout::Signature& shader_layout_signature)
  : device_(device),
    handle_(handle),
    shader_layout_signature_(shader_layout_signature),
    bindings_{} {
}

DescriptorSet::DescriptorSet(DescriptorSet&& other) noexcept
  : device_(other.device_),
    handle_(other.handle_),
    shader_layout_signature_(std::move(other.shader_layout_signature_)),
    bindings_(std::move(other.bindings_)) {
  other.handle_ = VK_NULL_HANDLE;
}

DescriptorSet& DescriptorSet::operator=(DescriptorSet&& other) noexcept {
  device_ = other.device_;
  handle_ = other.handle_;
  shader_layout_signature_ = std::move(other.shader_layout_signature_);
  bindings_ = std::move(other.bindings_);

  other.handle_ = VK_NULL_HANDLE;

  return *this;
}

DescriptorSet& DescriptorSet::bind(
    const uint32_t idx,
    const VulkanBuffer& buffer) {
  add_binding(DescriptorSet::ResourceBinding{
      idx,  // binding_idx
      shader_layout_signature_[idx],  // descriptor_type
      false,  // is_image
      {  // resource_info
        .buffer_info = {
          buffer.handle(),  // buffer
          buffer.mem_offset(),  // offset
          buffer.mem_range(),  // range
        },
      },
    });

  return *this;
}

DescriptorSet& DescriptorSet::bind(
    const uint32_t idx,
    const VulkanImage& image) {
  VkImageLayout binding_layout = image.layout();
  if (shader_layout_signature_[idx] == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
    binding_layout = VK_IMAGE_LAYOUT_GENERAL;
  }

  add_binding(DescriptorSet::ResourceBinding{
      idx,  // binding_idx
      shader_layout_signature_[idx],  // descriptor_type
      true,  // is_image
      {  // resource_info
        .image_info = {
          image.sampler(),  // buffer
          image.image_view(),  // imageView
          binding_layout,  // imageLayout
        },
      },
    });

  return *this;
}

VkDescriptorSet DescriptorSet::get_bind_handle() const {
  c10::SmallVector<VkWriteDescriptorSet, 6u> write_descriptor_sets;

  for (const ResourceBinding& binding : bindings_) {
    VkWriteDescriptorSet write{
      VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,  // sType
      nullptr,  // pNext
      handle_,  // dstSet
      binding.binding_idx,  // dstBinding
      0u,  // dstArrayElement
      1u,  // descriptorCount
      binding.descriptor_type,  // descriptorType
      nullptr,  // pImageInfo
      nullptr,  // pBufferInfo
      nullptr,  // pTexelBufferView
    };

    if (binding.is_image) {
      write.pImageInfo = &binding.resource_info.image_info;
    }
    else {
      write.pBufferInfo = &binding.resource_info.buffer_info;
    }

    write_descriptor_sets.emplace_back(write);
  }

  vkUpdateDescriptorSets(
      device_,
      write_descriptor_sets.size(),
      write_descriptor_sets.data(),
      0u,
      nullptr);

  VkDescriptorSet ret = handle_;

  return ret;
}

void DescriptorSet::add_binding(const ResourceBinding& binding) {
  const auto bindings_itr = std::find_if(
      bindings_.begin(),
      bindings_.end(),
      [binding_idx = binding.binding_idx](const ResourceBinding& other) {
        return other.binding_idx == binding_idx;
      });

  if (bindings_.end() == bindings_itr) {
    bindings_.emplace_back(binding);
  }
  else {
    *bindings_itr = binding;
  }
}

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

  std::vector<VkDescriptorSetLayout> descriptor_set_layouts(count);
  fill(
    descriptor_set_layouts.begin(),
    descriptor_set_layouts.end(),
    descriptor_set_layout
  );

  const VkDescriptorSetAllocateInfo descriptor_set_allocate_info{
    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    nullptr,
    descriptor_pool,
    utils::safe_downcast<uint32_t>(descriptor_set_layouts.size()),
    descriptor_set_layouts.data(),
  };

  VK_CHECK(vkAllocateDescriptorSets(
      device,
      &descriptor_set_allocate_info,
      descriptor_sets));
}

} // namespace

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

DescriptorSet Descriptor::Pool::allocate(
    const VkDescriptorSetLayout handle,
    const ShaderLayout::Signature& signature) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device_ && descriptor_pool_,
      "This descriptor pool is in an invalid state! "
      "Potential reason: This descriptor pool is moved from.");

  auto iterator = set_.layouts.find(handle);
  if (set_.layouts.cend() == iterator) {
    iterator = set_.layouts.insert({handle, {}}).first;
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
        handle,
        layout.pool.data() + layout.in_use,
        Configuration::kQuantum);
  }

  return DescriptorSet(
      device_,
      layout.pool[layout.in_use++],
      signature);
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
