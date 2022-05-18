#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Resource.h>
#include <ATen/native/vulkan/api/Shader.h>
#include <c10/util/hash.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

struct PipelineBarrier final {
  struct Stage final {
    VkPipelineStageFlags src;
    VkPipelineStageFlags dst;
  } stage;

  c10::SmallVector<Resource::Buffer::Barrier, 4u> buffers;
  c10::SmallVector<Resource::Image::Barrier, 4u> images;

  operator bool() const;
};

struct PipelineStage final {
  typedef uint8_t Flags;

  enum Type : Flags {
    None = 0u << 0u,
    Compute = 1u << 0u,
    Host = 1u << 1u,
    Transfer = 1u << 2u,
  };
};

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

  struct Hasher {
    size_t operator()(const VkDescriptorSetLayout) const;
  };

  // We need to define a custom swap function since this class
  // does not allow for move assignment. The swap function will
  // be used in the hash map.
  friend void swap(PipelineLayout& lhs, PipelineLayout& rhs);
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

  struct Hasher {
    size_t operator()(const Descriptor&) const;
  };

  // We need to define a custom swap function since this class
  // does not allow for move assignment. The swap function will
  // be used in the hash map.
  friend void swap(ComputePipeline& lhs, ComputePipeline& rhs);
};

class PipelineLayoutCache final {
 public:
  explicit PipelineLayoutCache(const VkDevice device);

  PipelineLayoutCache(const PipelineLayoutCache&) = delete;
  PipelineLayoutCache& operator=(const PipelineLayoutCache&) = delete;

  PipelineLayoutCache(PipelineLayoutCache&&) = default;
  PipelineLayoutCache& operator=(PipelineLayoutCache&&) = delete;

  ~PipelineLayoutCache();

  typedef VkDescriptorSetLayout Key;
  typedef PipelineLayout Value;
  typedef PipelineLayout::Hasher Hasher;

 private:
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

  ComputePipelineCache(ComputePipelineCache&&) = default;
  ComputePipelineCache& operator=(ComputePipelineCache&&) = delete;

  ~ComputePipelineCache();

  typedef ComputePipeline::Descriptor Key;
  typedef ComputePipeline Value;
  typedef ComputePipeline::Hasher Hasher;

 private:
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
