#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/api.h>
#include <ATen/native/vulkan/impl/Registry.h>

#define VK_KERNEL(shader_name) \
  ::at::native::vulkan::get_shader_info(#shader_name)
#define VK_LOOKUP_KERNEL(op_name) \
  ::at::native::vulkan::look_up_shader_info(#op_name)

namespace at {
namespace native {
namespace vulkan {

/*
 * Maps a semantic dimension name to an integer that corresponds to its
 * innermost ordering in a 4D tensor in NCHW format. Width is the innermost
 * dimension, so it corresponds to 1, height is the next innermost, so it
 * corresponds to 2, and so on.
 */
struct Dim4D {
  static constexpr uint32_t Width = 1u;
  static constexpr uint32_t Height = 2u;
  static constexpr uint32_t Channel = 3u;
  static constexpr uint32_t Batch = 4u;
};

/*
 * Semantic dimension names for a 1D tensor
 */
struct Dim1D {
  static constexpr uint32_t Length = 1u;
};

/*
 * Semantic dimension names for a 2D Convolution kernel.
 */
struct DimConv2DKernel {
  static constexpr uint32_t Width = 1u;
  static constexpr uint32_t Height = 2u;
  static constexpr uint32_t InChannels = 3u;
  static constexpr uint32_t OutChannels = 4u;
};

/*
 * The same as the above, except for a 2D Transposed Convolution kernel.
 */
struct DimTConv2DKernel {
  static constexpr uint32_t Width = 1u;
  static constexpr uint32_t Height = 2u;
  static constexpr uint32_t OutChannels = 3u;
  static constexpr uint32_t InChannels = 4u;
};

/*
 * The functions below safely return the size of the dimension at the N-th
 * innermost index. If the dimensionality of the size array is not sufficient
 * then 1 will be returned. The structs above are intended to be used with
 * these functions.
 */
template <uint32_t N>
uint32_t dim_at(const std::vector<int64_t>& sizes) {
  const uint32_t dims = sizes.size();
  return dims < N ? 1 : api::utils::safe_downcast<uint32_t>(sizes[dims - N]);
}

template <uint32_t N>
uint32_t dim_at(const vTensor& v_in) {
  return dim_at<N>(v_in.sizes());
}

/*
 * For most global work group sizes, returns {4, 4, 4}, but adjusts the size for
 * 2D global work group sizes. Always maintains a total of 64 invocations
 */
api::utils::uvec3 adaptive_work_group_size(
    const api::utils::uvec3& global_work_group);

} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
