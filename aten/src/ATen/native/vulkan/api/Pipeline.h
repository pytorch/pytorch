#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Resource.h>
#include <ATen/native/vulkan/api/Shader.h>
#include <c10/util/SmallVector.h>
#include <c10/util/flat_hash_map.h>

#include <mutex>

namespace at {
namespace native {
namespace vulkan {
namespace api {

struct PipelineBarrier final {
  struct Stages final {
    VkPipelineStageFlags src;
    VkPipelineStageFlags dst;
  } stage;

  c10::SmallVector<BufferMemoryBarrier, 4u> buffers;
  c10::SmallVector<ImageMemoryBarrier, 4u> images;

  inline operator bool() const {
    return (0u != stage.src) || (0u != stage.dst) || !buffers.empty() ||
        !images.empty();
  }
};

using PipelineStageFlags = uint8_t;

enum PipelineStage : PipelineStageFlags {
  NO_STAGE = 0u << 0u,
  COMPUTE = 1u << 0u,
  HOST = 1u << 1u,
  TRANSFER = 1u << 2u,
};

VkAccessFlags vk_access(const PipelineStageFlags, const MemoryAccessFlags);
VkPipelineStageFlags vk_stage(const PipelineStageFlags);
VkImageLayout vk_layout(const PipelineStageFlags, const MemoryAccessFlags);

class PipelineLayout final {
 public:
  explicit PipelineLayout(const VkDevice, const VkDescriptorSetLayout);

  PipelineLayout(const PipelineLayout&) = delete;
  PipelineLayout& operator=(const PipelineLayout&) = delete;

  PipelineLayout(PipelineLayout&&) noexcept;
  PipelineLayout& operator=(PipelineLayout&&) = delete;

  ~PipelineLayout();

 private:
  VkDevice device_;
  VkPipelineLayout handle_;

 public:
  VkPipelineLayout handle() const {
    return handle_;
  }

  // We need to define a custom swap function since this class
  // does not allow for move assignment. The swap function will
  // be used in the hash map.
  friend void swap(PipelineLayout& lhs, PipelineLayout& rhs) noexcept;
};

class ComputePipeline final {
 public:
  struct Descriptor final {
    VkPipelineLayout pipeline_layout;
    VkShaderModule shader_module;
    utils::uvec3 local_work_group;
  };

  explicit ComputePipeline(
      const VkDevice device,
      const Descriptor& descriptor,
      const VkPipelineCache pipeline_cache);

  ComputePipeline(const ComputePipeline&) = delete;
  ComputePipeline& operator=(const ComputePipeline&) = delete;

  ComputePipeline(ComputePipeline&&) noexcept;
  ComputePipeline& operator=(ComputePipeline&&) = delete;

  ~ComputePipeline();

 private:
  VkDevice device_;
  VkPipeline handle_;

 public:
  inline VkPipeline handle() const {
    return handle_;
  }

  // We need to define a custom swap function since this class
  // does not allow for move assignment. The swap function will
  // be used in the hash map.
  friend void swap(ComputePipeline& lhs, ComputePipeline& rhs) noexcept;
};

class PipelineLayoutCache final {
 public:
  explicit PipelineLayoutCache(const VkDevice device);

  PipelineLayoutCache(const PipelineLayoutCache&) = delete;
  PipelineLayoutCache& operator=(const PipelineLayoutCache&) = delete;

  PipelineLayoutCache(PipelineLayoutCache&&) noexcept;
  PipelineLayoutCache& operator=(PipelineLayoutCache&&) = delete;

  ~PipelineLayoutCache();

  using Key = VkDescriptorSetLayout;
  using Value = PipelineLayout;

  struct Hasher {
    inline size_t operator()(
        const VkDescriptorSetLayout descriptor_layout) const {
      return c10::get_hash(descriptor_layout);
    }
  };

 private:
  // Multiple threads could potentially be adding entries into the cache, so use
  // a mutex to manage access
  std::mutex cache_mutex_;

  VkDevice device_;
  ska::flat_hash_map<Key, Value, Hasher> cache_;

 public:
  VkPipelineLayout retrieve(const Key&);
  void purge();
};

class ComputePipelineCache final {
 public:
  explicit ComputePipelineCache(const VkDevice device);

  ComputePipelineCache(const ComputePipelineCache&) = delete;
  ComputePipelineCache& operator=(const ComputePipelineCache&) = delete;

  ComputePipelineCache(ComputePipelineCache&&) noexcept;
  ComputePipelineCache& operator=(ComputePipelineCache&&) = delete;

  ~ComputePipelineCache();

  using Key = ComputePipeline::Descriptor;
  using Value = ComputePipeline;

  struct Hasher {
    inline size_t operator()(
        const ComputePipeline::Descriptor& descriptor) const {
      return c10::get_hash(
          descriptor.pipeline_layout,
          descriptor.shader_module,
          descriptor.local_work_group.data[0u],
          descriptor.local_work_group.data[1u],
          descriptor.local_work_group.data[2u]);
    };
  };

 private:
  // Multiple threads could potentially be adding entries into the cache, so use
  // a mutex to manage access
  std::mutex cache_mutex_;

  VkDevice device_;
  VkPipelineCache pipeline_cache_;
  ska::flat_hash_map<Key, Value, Hasher> cache_;

 public:
  VkPipeline retrieve(const Key&);
  void purge();
};

//
// Impl
//

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
