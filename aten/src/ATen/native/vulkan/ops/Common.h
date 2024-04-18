#pragma once

#ifdef USE_VULKAN_API

#include <c10/util/ArrayRef.h>

#include <ATen/core/List.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/vulkan/api/api.h>
#include <ATen/native/vulkan/impl/Common.h>
#include <ATen/native/vulkan/ops/Convert.h>

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

  // Parameters (Pooling Kernels, Dilation, Padding, Stride, etc.)
  struct BatchMatrices final {
    static constexpr size_t batch = 0u;
    static constexpr size_t height = 1u;
    static constexpr size_t width = 2u;
  };
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

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
