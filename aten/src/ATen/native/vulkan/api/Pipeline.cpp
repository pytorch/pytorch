#include <ATen/native/vulkan/api/Pipeline.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

//
// Utility Functions
//

VkAccessFlags vk_access(
    const PipelineStageFlags stage,
    const MemoryAccessFlags access) {
  VkAccessFlags vk_access = 0u;

  if (access & MemoryAccessType::READ) {
    if (stage & PipelineStage::COMPUTE) {
      vk_access |= VK_ACCESS_SHADER_READ_BIT;
    }

    if (stage & PipelineStage::HOST) {
      vk_access |= VK_ACCESS_HOST_READ_BIT;
    }

    if (stage & PipelineStage::TRANSFER) {
      vk_access |= VK_ACCESS_TRANSFER_READ_BIT;
    }
  }

  if (access & MemoryAccessType::WRITE) {
    if (stage & PipelineStage::COMPUTE) {
      vk_access |= VK_ACCESS_SHADER_WRITE_BIT;
    }

    if (stage & PipelineStage::HOST) {
      vk_access |= VK_ACCESS_HOST_WRITE_BIT;
    }

    if (stage & PipelineStage::TRANSFER) {
      vk_access |= VK_ACCESS_TRANSFER_WRITE_BIT;
    }
  }

  return vk_access;
}

VkPipelineStageFlags vk_stage(const PipelineStageFlags stage) {
  VkPipelineStageFlags vk_stage = 0u;

  if (stage & PipelineStage::COMPUTE) {
    vk_stage |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  }

  if (stage & PipelineStage::HOST) {
    vk_stage |= VK_PIPELINE_STAGE_HOST_BIT;
  }

  if (stage & PipelineStage::TRANSFER) {
    vk_stage |= VK_PIPELINE_STAGE_TRANSFER_BIT;
  }

  return vk_stage;
}

VkImageLayout vk_layout(
    const PipelineStageFlags stage,
    const MemoryAccessFlags access) {
  switch (stage) {
    case PipelineStage::COMPUTE:
      switch (access) {
        case MemoryAccessType::READ:
          return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        default:
          return VK_IMAGE_LAYOUT_GENERAL;
      }
      break;
    case PipelineStage::TRANSFER:
      switch (access) {
        case MemoryAccessType::READ:
          return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        case MemoryAccessType::WRITE:
          return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        default:
          VK_THROW("Invalid memory access type for transfer stage!");
      }
      break;
    default:
      VK_THROW("Cannot determine appropriate image layout");
  }

  return VK_IMAGE_LAYOUT_UNDEFINED;
}

//
// PipelineLayout
//

PipelineLayout::PipelineLayout(
    VkDevice device,
    VkDescriptorSetLayout descriptor_layout)
    : device_(device), handle_{VK_NULL_HANDLE} {
  // TODO: Enable push constants
  const VkPipelineLayoutCreateInfo pipeline_layout_create_info{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      1u, // setLayoutCount
      &descriptor_layout, // pSetLayouts
      0u, // pushConstantRangeCount
      nullptr, // pPushConstantRanges
  };

  VK_CHECK(vkCreatePipelineLayout(
      device_, &pipeline_layout_create_info, nullptr, &handle_));
}

PipelineLayout::PipelineLayout(PipelineLayout&& other) noexcept
    : device_(other.device_), handle_(other.handle_) {
  other.handle_ = VK_NULL_HANDLE;
}

PipelineLayout::~PipelineLayout() {
  if (VK_NULL_HANDLE == handle_) {
    return;
  }
  vkDestroyPipelineLayout(device_, handle_, nullptr);
  handle_ = VK_NULL_HANDLE;
}

void swap(PipelineLayout& lhs, PipelineLayout& rhs) noexcept {
  VkDevice tmp_device = lhs.device_;
  VkPipelineLayout tmp_handle = lhs.handle_;

  lhs.device_ = rhs.device_;
  lhs.handle_ = rhs.handle_;

  rhs.device_ = tmp_device;
  rhs.handle_ = tmp_handle;
}

//
// ComputePipeline
//

ComputePipeline::ComputePipeline(
    VkDevice device,
    const ComputePipeline::Descriptor& descriptor,
    VkPipelineCache pipeline_cache)
    : device_(device), handle_{VK_NULL_HANDLE} {
  // NOLINTNEXTLINE
  constexpr VkSpecializationMapEntry specialization_map_entries[3]{
      // X
      {
          0u,
          offsetof(utils::uvec3, data[0u]),
          sizeof(utils::uvec3::data[0u]),
      },
      // Y
      {
          1u,
          offsetof(utils::uvec3, data[1u]),
          sizeof(utils::uvec3::data[1u]),
      },
      // Z
      {
          2u,
          offsetof(utils::uvec3, data[2u]),
          sizeof(utils::uvec3::data[2u]),
      },
  };

  const VkSpecializationInfo specialization_info{
      3u, // mapEntryCount
      specialization_map_entries, // pMapEntries
      sizeof(descriptor.local_work_group), // dataSize
      &descriptor.local_work_group, // pData
  };

  const VkPipelineShaderStageCreateInfo shader_stage_create_info{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      VK_SHADER_STAGE_COMPUTE_BIT, // stage
      descriptor.shader_module, // module
      "main", // pName
      &specialization_info, // pSpecializationInfo
  };

  const VkComputePipelineCreateInfo compute_pipeline_create_info{
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      shader_stage_create_info, // stage
      descriptor.pipeline_layout, // layout
      VK_NULL_HANDLE, // basePipelineHandle
      0u, // basePipelineIndex
  };

  VK_CHECK(vkCreateComputePipelines(
      device_,
      pipeline_cache,
      1u,
      &compute_pipeline_create_info,
      nullptr,
      &handle_));
}

ComputePipeline::ComputePipeline(ComputePipeline&& other) noexcept
    : device_(other.device_), handle_(other.handle_) {
  other.handle_ = VK_NULL_HANDLE;
}

ComputePipeline::~ComputePipeline() {
  if (VK_NULL_HANDLE == handle_) {
    return;
  }
  vkDestroyPipeline(device_, handle_, nullptr);
  handle_ = VK_NULL_HANDLE;
}

void swap(ComputePipeline& lhs, ComputePipeline& rhs) noexcept {
  VkDevice tmp_device = lhs.device_;
  VkPipeline tmp_handle = lhs.handle_;

  lhs.device_ = rhs.device_;
  lhs.handle_ = rhs.handle_;

  rhs.device_ = tmp_device;
  rhs.handle_ = tmp_handle;
}

bool operator==(
    const ComputePipeline::Descriptor& _1,
    const ComputePipeline::Descriptor& _2) {
  return (
      _1.pipeline_layout == _2.pipeline_layout &&
      _1.shader_module == _2.shader_module &&
      _1.local_work_group == _2.local_work_group);
}

//
// PipelineLayoutCache
//

PipelineLayoutCache::PipelineLayoutCache(VkDevice device)
    : cache_mutex_{}, device_(device), cache_{} {}

PipelineLayoutCache::PipelineLayoutCache(PipelineLayoutCache&& other) noexcept
    : cache_mutex_{}, device_(other.device_), cache_(std::move(other.cache_)) {
  std::lock_guard<std::mutex> lock(other.cache_mutex_);
}

PipelineLayoutCache::~PipelineLayoutCache() {
  purge();
}

VkPipelineLayout PipelineLayoutCache::retrieve(
    const PipelineLayoutCache::Key& key) {
  std::lock_guard<std::mutex> lock(cache_mutex_);

  auto it = cache_.find(key);
  if (cache_.cend() == it) {
    it = cache_.insert({key, PipelineLayoutCache::Value(device_, key)}).first;
  }

  return it->second.handle();
}

void PipelineLayoutCache::purge() {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  cache_.clear();
}

//
// ComputePipelineCache
//

ComputePipelineCache::ComputePipelineCache(VkDevice device)
    : cache_mutex_{},
      device_(device),
      pipeline_cache_{VK_NULL_HANDLE},
      cache_{} {
  const VkPipelineCacheCreateInfo pipeline_cache_create_info{
      VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      0u, // initialDataSize
      nullptr, // pInitialData
  };

  VK_CHECK(vkCreatePipelineCache(
      device, &pipeline_cache_create_info, nullptr, &pipeline_cache_));
}

ComputePipelineCache::ComputePipelineCache(
    ComputePipelineCache&& other) noexcept
    : cache_mutex_{},
      device_(other.device_),
      pipeline_cache_(other.pipeline_cache_),
      cache_(std::move(other.cache_)) {
  std::lock_guard<std::mutex> lock(other.cache_mutex_);

  other.pipeline_cache_ = VK_NULL_HANDLE;
}

ComputePipelineCache::~ComputePipelineCache() {
  purge();

  if (VK_NULL_HANDLE == pipeline_cache_) {
    return;
  }
  vkDestroyPipelineCache(device_, pipeline_cache_, nullptr);
  pipeline_cache_ = VK_NULL_HANDLE;
}

VkPipeline ComputePipelineCache::retrieve(
    const ComputePipelineCache::Key& key) {
  std::lock_guard<std::mutex> lock(cache_mutex_);

  auto it = cache_.find(key);
  if (cache_.cend() == it) {
    it = cache_
             .insert(
                 {key,
                  ComputePipelineCache::Value(device_, key, pipeline_cache_)})
             .first;
  }

  return it->second.handle();
}

void ComputePipelineCache::purge() {
  cache_.clear();
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
