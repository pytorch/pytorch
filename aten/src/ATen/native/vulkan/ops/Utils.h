#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

namespace utils {

Tensor nchw_to_nc4hw(const Tensor&);

Tensor create_staging_tensor(const vTensor&);

Tensor nc4hw_to_nchw(const Tensor&, IntArrayRef);

void copy_buffer_to_buffer(
    api::Context* const context,
    api::StorageBuffer& src,
    api::StorageBuffer& dst,
    VkFence fence_handle);

void copy_buffer_to_vtensor(
    api::VulkanBuffer&,
    vTensor&,
    api::PipelineBarrier&);

void copy_vtensor_to_buffer(
    vTensor&,
    api::VulkanBuffer&,
    api::PipelineBarrier&,
    const VkFence fence_handle = VK_NULL_HANDLE);

inline int64_t normalize(const int64_t dimension, const int64_t n) {
  return (dimension % n + n) % n;
}

void pack_buffer_to_vtensor(
    api::VulkanBuffer&,
    vTensor&,
    api::PipelineBarrier&);

void pack_staging_to_vtensor(api::VulkanBuffer&, vTensor&);

bool pack_vtensor_to_staging(
    vTensor&,
    api::VulkanBuffer&,
    const VkFence fence_handle = VK_NULL_HANDLE);

// Broadcasting Utils
void is_broadcastable(const Tensor& input1, const Tensor& input2);
std::vector<int64_t> broadcast_size(const Tensor& t1, const Tensor& t2);

// This function returns the value of the underlying texel at pos of the given
// tensor. It is useful for debugging and unit test at which we want to verify
// the actual tensor layout. This function is very slow as it involves a fench
// to extract just one value.
api::utils::vec4 extract_texel(
    const Tensor& tensor,
    const api::utils::ivec3& pos);

inline api::utils::ivec2 make_ivec2(
    const IntArrayRef ints,
    bool reverse = false) {
  VK_CHECK_COND(ints.size() == 2);
  if (reverse) {
    return {
        api::utils::safe_downcast<int32_t>(ints[1]),
        api::utils::safe_downcast<int32_t>(ints[0])};
  } else {
    return {
        api::utils::safe_downcast<int32_t>(ints[0]),
        api::utils::safe_downcast<int32_t>(ints[1])};
  }
}

inline api::utils::ivec4 make_ivec4(
    const IntArrayRef ints,
    bool reverse = false) {
  VK_CHECK_COND(ints.size() == 4);
  if (reverse) {
    return {
        api::utils::safe_downcast<int32_t>(ints[3]),
        api::utils::safe_downcast<int32_t>(ints[2]),
        api::utils::safe_downcast<int32_t>(ints[1]),
        api::utils::safe_downcast<int32_t>(ints[0]),
    };
  } else {
    return {
        api::utils::safe_downcast<int32_t>(ints[0]),
        api::utils::safe_downcast<int32_t>(ints[1]),
        api::utils::safe_downcast<int32_t>(ints[2]),
        api::utils::safe_downcast<int32_t>(ints[3]),
    };
  }
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
