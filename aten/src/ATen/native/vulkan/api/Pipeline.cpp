#include <ATen/native/vulkan/api/Pipeline.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

//
// PipelineBarrier
//

PipelineBarrier::operator bool() const {
  return (0u != stage.src) ||
         (0u != stage.dst) ||
         !buffers.empty() ||
         !images.empty();
}

//
// PipelineLayout
//

PipelineLayout::PipelineLayout(
    const VkDevice device,
    const VkDescriptorSetLayout descriptor_layout)
  : device_(device) {
  // TODO: Enable push constants
  const VkPipelineLayoutCreateInfo pipeline_layout_create_info{
    VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,  // sType
    nullptr,  // pNext
    0u,  // flags
    1u,  // setLayoutCount
    &descriptor_layout,  // pSetLayouts
    0u,  // pushConstantRangeCount
    nullptr,  // pPushConstantRanges
  };

  VK_CHECK(vkCreatePipelineLayout(
      device_,
      &pipeline_layout_create_info,
      nullptr,
      &handle_));
}

PipelineLayout::PipelineLayout(PipelineLayout&& other) noexcept
  : device_(other.device_),
    handle_(other.handle_) {
  other.handle_ = VK_NULL_HANDLE;
}

PipelineLayout::~PipelineLayout() {
  if C10_LIKELY(VK_NULL_HANDLE == handle_) {
    return;
  }
  vkDestroyPipelineLayout(device_, handle_, nullptr);
  handle_ = VK_NULL_HANDLE;
}

void swap(PipelineLayout& lhs, PipelineLayout& rhs) {
  VkDevice tmp_device = lhs.device_;
  VkPipelineLayout tmp_handle = lhs.handle_;

  lhs.device_ = rhs.device_;
  lhs.handle_ = rhs.handle_;

  rhs.device_ = tmp_device;
  rhs.handle_ = tmp_handle;
}

size_t PipelineLayout::Hasher::operator()(
    const VkDescriptorSetLayout descriptor_layout) const {
  return c10::get_hash(descriptor_layout);
}

//
// ComputePipeline
//

ComputePipeline::ComputePipeline(
    const VkDevice device,
    const ComputePipeline::Descriptor& descriptor,
    const VkPipelineCache pipeline_cache)
  : device_(device) {
  constexpr VkSpecializationMapEntry specialization_map_entires[3]{
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
    3u,  // mapEntryCount
    specialization_map_entires,  // pMapEntries
    sizeof(descriptor.local_work_group),  // dataSize
    &descriptor.local_work_group,  // pData
  };

  const VkPipelineShaderStageCreateInfo shader_stage_create_info{
    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,  // sType
    nullptr,  // pNext
    0u,  // flags
    VK_SHADER_STAGE_COMPUTE_BIT,  // stage
    descriptor.shader_module,  // module
    "main",  // pName
    &specialization_info,  // pSpecializationInfo
  };

  const VkComputePipelineCreateInfo compute_pipeline_create_info{
    VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,  // sType
    nullptr,  // pNext
    0u,  // flags
    shader_stage_create_info,  // stage
    descriptor.pipeline_layout,  // layout
    VK_NULL_HANDLE,  // basePipelineHandle
    0u,  // basePipelineIndex
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
  : device_(other.device_),
    handle_(other.handle_) {
  other.handle_ = VK_NULL_HANDLE;
}

ComputePipeline::~ComputePipeline() {
  if C10_LIKELY(VK_NULL_HANDLE == handle_) {
    return;
  }
  vkDestroyPipeline(device_, handle_, nullptr);
  handle_ = VK_NULL_HANDLE;
}

void swap(ComputePipeline& lhs, ComputePipeline& rhs) {
  VkDevice tmp_device = lhs.device_;
  VkPipeline tmp_handle = lhs.handle_;

  lhs.device_ = rhs.device_;
  lhs.handle_ = rhs.handle_;

  rhs.device_ = tmp_device;
  rhs.handle_ = tmp_handle;
}

size_t ComputePipeline::Hasher::operator()(
    const ComputePipeline::Descriptor& descriptor) const {
  return c10::get_hash(
      descriptor.pipeline_layout,
      descriptor.shader_module,
      descriptor.local_work_group.data[0u],
      descriptor.local_work_group.data[1u],
      descriptor.local_work_group.data[2u]);
}

bool operator==(
    const ComputePipeline::Descriptor& _1,
    const ComputePipeline::Descriptor& _2) {

  return (_1.pipeline_layout == _2.pipeline_layout && \
          _1.shader_module == _2.shader_module && \
          _1.local_work_group == _2.local_work_group);
}

//
// PipelineLayoutCache
//

PipelineLayoutCache::PipelineLayoutCache(const VkDevice device)
  : device_(device),
    cache_{} {
}

PipelineLayoutCache::~PipelineLayoutCache() {
  purge();
}

VkPipelineLayout PipelineLayoutCache::retrieve(
    const PipelineLayoutCache::Key& key) {
  auto it = cache_.find(key);
  if C10_UNLIKELY(cache_.cend() == it) {
    it = cache_.insert({key, PipelineLayoutCache::Value(device_, key)}).first;
  }

  return it->second.handle();
}

void PipelineLayoutCache::purge() {
  cache_.clear();
}

//
// ComputePipelineCache
//

ComputePipelineCache::ComputePipelineCache(const VkDevice device)
  : device_(device),
    cache_{} {
  const VkPipelineCacheCreateInfo pipeline_cache_create_info{
    VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,  // sType
    nullptr,  // pNext
    0u,  // flags
    0u,  // initialDataSize
    nullptr,  // pInitialData
  };

  VK_CHECK(vkCreatePipelineCache(
      device,
      &pipeline_cache_create_info,
      nullptr,
      &pipeline_cache_));
}

ComputePipelineCache::~ComputePipelineCache() {
  purge();
}

VkPipeline ComputePipelineCache::retrieve(
    const ComputePipelineCache::Key& key) {
  auto it = cache_.find(key);
  if C10_UNLIKELY(cache_.cend() == it) {
    it = cache_.insert(
        {key, ComputePipelineCache::Value(device_, key, pipeline_cache_)}).first;
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
