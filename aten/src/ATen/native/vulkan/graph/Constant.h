#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Context.h>

namespace at {
namespace native {
namespace vulkan {

/*
 * Represents a reference to a tensor that has been serialized with the model,
 * such as a serialized weight tensor. It contains some metadata as well as a
 * raw pointer to the data of the tensor, which is assumed to be contiguous.
 */
struct TensorRef final {
  std::vector<int64_t> sizes;
  api::ScalarType dtype;
  const void* data;

  explicit TensorRef(
      const std::vector<int64_t>& t_sizes,
      api::ScalarType t_dtype,
      const void* const t_data);

  TensorRef(const TensorRef&) = default;
  TensorRef& operator=(const TensorRef&) = default;

  TensorRef(TensorRef&&) = default;
  TensorRef& operator=(TensorRef&&) = default;
};

} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
