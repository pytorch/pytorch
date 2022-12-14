#pragma once

#ifdef USE_VULKAN_API

#include <ATen/core/List.h>
#include <ATen/core/Tensor.h>
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
uint32_t get_dim(const IntArrayRef sizes) {
  const uint32_t dims = sizes.size();
  return dims < N ? 1 : api::utils::safe_downcast<uint32_t>(sizes[dims - N]);
}

template <uint32_t N>
uint32_t get_dim(const Tensor& t_in) {
  return get_dim<N>(t_in.sizes());
}

template <uint32_t N>
uint32_t get_dim(const vTensor& v_in) {
  return get_dim<N>(v_in.sizes());
}

/*
 * Given an IntArrayRef of up to 4 elements, constructs a uvec4 containing those
 * elements in reverse order.
 */
api::utils::uvec4 make_nchw_uvec4(const IntArrayRef arr);

inline c10::optional<Tensor> get_optional_tensor(
    const c10::impl::GenericList& gen_list,
    const uint32_t idx) {
  return gen_list.get(idx).isTensor() ? gen_list.get(idx).toTensor()
                                      : c10::optional<Tensor>();
}

inline c10::optional<Scalar> get_optional_scalar(
    const c10::impl::GenericList& gen_list,
    const uint32_t idx) {
  return gen_list.get(idx).isScalar() ? gen_list.get(idx).toScalar()
                                      : c10::optional<Scalar>();
}

api::utils::uvec3 adaptive_work_group_size(
    const api::utils::uvec3& global_work_group);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
