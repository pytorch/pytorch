#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Resource.h>
#include <ATen/native/vulkan/api/Shader.h>
#include <c10/util/flat_hash_map.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

class DescriptorSet final {
 public:
  explicit DescriptorSet(
      const VkDevice,
      const VkDescriptorSet,
      const ShaderLayout::Signature&);

  DescriptorSet(const DescriptorSet&) = delete;
  DescriptorSet& operator=(const DescriptorSet&) = delete;

  DescriptorSet(DescriptorSet&&) noexcept;
  DescriptorSet& operator=(DescriptorSet&&) noexcept;

  ~DescriptorSet() = default;

  struct ResourceBinding final {
    uint32_t binding_idx;
    VkDescriptorType descriptor_type;
    bool is_image;

    union {
      VkDescriptorBufferInfo buffer_info;
      VkDescriptorImageInfo image_info;
    } resource_info;
  };

 private:
  VkDevice device_;
  VkDescriptorSet handle_;
  ShaderLayout::Signature shader_layout_signature_;
  c10::SmallVector<ResourceBinding, 6u> bindings_;

 public:
  DescriptorSet& bind(const uint32_t, const VulkanBuffer&);
  DescriptorSet& bind(const uint32_t, const VulkanImage&);

  VkDescriptorSet get_bind_handle() const;

 private:
  void add_binding(const ResourceBinding& resource);
};

class DescriptorSetPile final {
 public:
  DescriptorSetPile(
      const uint32_t,
      const VkDescriptorSetLayout,
      const VkDevice,
      const VkDescriptorPool);

  DescriptorSetPile(const DescriptorSetPile&) = delete;
  DescriptorSetPile& operator=(const DescriptorSetPile&) = delete;

  DescriptorSetPile(DescriptorSetPile&&) = default;
  DescriptorSetPile& operator=(DescriptorSetPile&&) = default;

  ~DescriptorSetPile() = default;

 private:
  uint32_t pile_size_;
  VkDescriptorSetLayout set_layout_;
  VkDevice device_;
  VkDescriptorPool pool_;
  std::vector<VkDescriptorSet> descriptors_;
  size_t in_use_;

 public:
  VkDescriptorSet get_descriptor_set();

 private:
  void allocate_new_batch();
};

struct DescriptorPoolConfig final {
  // Overall Pool capacity
  uint32_t descriptorPoolMaxSets;
  // DescriptorCounts by type
  uint32_t descriptorUniformBufferCount;
  uint32_t descriptorStorageBufferCount;
  uint32_t descriptorCombinedSamplerCount;
  uint32_t descriptorStorageImageCount;
  // Pile size for pre-allocating descriptor sets
  uint32_t descriptorPileSizes;
};

class DescriptorPool final {
 public:
  explicit DescriptorPool(const VkDevice, const DescriptorPoolConfig&);

  DescriptorPool(const DescriptorPool&) = delete;
  DescriptorPool& operator=(const DescriptorPool&) = delete;

  DescriptorPool(DescriptorPool&&) = delete;
  DescriptorPool& operator=(DescriptorPool&&) = delete;

  ~DescriptorPool();

 private:
  VkDevice device_;
  VkDescriptorPool pool_;
  DescriptorPoolConfig config_;
  // New Descriptors
  std::mutex mutex_;
  ska::flat_hash_map<VkDescriptorSetLayout, DescriptorSetPile> piles_;

 public:
  DescriptorSet get_descriptor_set(
      const VkDescriptorSetLayout handle,
      const ShaderLayout::Signature& signature);

  void flush();
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
