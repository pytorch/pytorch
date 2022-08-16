#pragma once

#ifdef USE_VULKAN_API

#include <ATen/ATen.h>
#include <ATen/native/vulkan/api/api.h>
#include <ATen/native/vulkan/ops/Tensor.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

struct Layout final {
  // 4D Activation Maps
  struct Activation4D final {
    static constexpr size_t batch = 0u;
    static constexpr size_t channels = 1u;
    static constexpr size_t height = 2u;
    static constexpr size_t width = 3u;
  };

  // Convolution Filters
  struct Filter final {
    static constexpr size_t output = 0u;
    static constexpr size_t input = 1u;
    static constexpr size_t height = 2u;
    static constexpr size_t width = 3u;
  };

  // Transposed Convolution Filters
  struct TransposedFilter final {
    static constexpr size_t input = 0u;
    static constexpr size_t output = 1u;
    static constexpr size_t height = 2u;
    static constexpr size_t width = 3u;
  };

  // Parameters (Pooling Kernels, Dilation, Padding, Stride, etc.)
  struct Parameter final {
    static constexpr size_t height = 0u;
    static constexpr size_t width = 1u;
  };
};

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
 * The functions below safely return the size of the dimension at the N-th
 * innermost index. If the dimensionality of the size array is not sufficient
 * then 1 will be returned. The structs above are intended to be used with
 * these functions.
 */
template <uint32_t N>
uint32_t get_dim(const IntArrayRef sizes) {
  const uint32_t dims = sizes.size();
  return dims < N ? 1 : sizes[dims - N];
}

template <uint32_t N>
uint32_t get_dim(const Tensor& t_in) {
  return get_dim<N>(t_in.sizes());
}

template <uint32_t N>
uint32_t get_dim(const vTensor& v_in) {
  return get_dim<N>(v_in.sizes());
}

api::utils::uvec3 adaptive_work_group_size(
    const api::utils::uvec3& global_work_group);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
