#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

inline int64_t normalize(const int64_t dimension, const int64_t n) {
  return (dimension % n + n) % n;
}

void pack_buffer_to_vtensor(
    api::VulkanBuffer&,
    vTensor&,
    api::PipelineBarrier&);

void pack_staging_to_vtensor(api::VulkanBuffer&, vTensor&);

void pack_vtensor_to_staging(
    vTensor&,
    api::VulkanBuffer&,
    const VkFence fence_handle = VK_NULL_HANDLE);

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
