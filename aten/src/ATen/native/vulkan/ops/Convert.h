#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/VulkanOpaqueTensorImpl.h>
#include <ATen/native/vulkan/api/Tensor.h>
#include <ATen/native/vulkan/api/Types.h>
#include <c10/util/accumulate.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

/*
 * Converts a `c10::ScalarType` to an equivalent
 * `::at::native::vulkan::api::ScalarType`.
 */
static inline api::ScalarType convert_dtype(const c10::ScalarType dtype) {
#define DEFINE_CASE(ctype, vkformat, name) \
  case c10::ScalarType::name:              \
    return ::at::native::vulkan::api::ScalarType::name;

  switch (dtype) {
    VK_FORALL_SCALAR_TYPES(DEFINE_CASE)
    default:
      TORCH_CHECK(false, "Not a supported Vulkan ScalarType!");
  }
#undef DEFINE_CASE
}

/*
 * Converts an `::at::native::vulkan::api::ScalarType` to an equivalent
 * `c10::ScalarType`.
 */
static inline c10::ScalarType convert_dtype(const api::ScalarType dtype) {
#define DEFINE_CASE(ctype, vkformat, name)          \
  case ::at::native::vulkan::api::ScalarType::name: \
    return c10::ScalarType::name;

  switch (dtype) {
    VK_FORALL_SCALAR_TYPES(DEFINE_CASE)
    default:
      TORCH_CHECK(false, "Not a supported c10::ScalarType!");
  }
#undef DEFINE_CASE
}

using vTensorImpl = VulkanOpaqueTensorImpl<vTensor>;

inline Tensor convert(const vTensor& tensor) {
  return at::detail::make_tensor<vTensorImpl>(
      DispatchKeySet(DispatchKey::Vulkan),
      c10::scalarTypeToTypeMeta(convert_dtype(tensor.dtype())),
      at::Device(at::kVulkan),
      tensor,
      tensor.sizes(),
      tensor.strides());
}

inline Tensor convert_quantized(const vTensor& tensor) {
  TORCH_CHECK(tensor.is_quantized(), "Not a Quantized Tensor");
  return at::detail::make_tensor<vTensorImpl>(
      DispatchKeySet(DispatchKey::Vulkan),
      c10::scalarTypeToTypeMeta(convert_dtype(tensor.dtype())),
      at::Device(at::kVulkan),
      tensor,
      tensor.sizes(),
      tensor.strides());
}

inline vTensor& convert(const Tensor& tensor) {
  TORCH_INTERNAL_ASSERT(tensor.is_vulkan(), "Vulkan tensor expected!");

  vTensorImpl* const impl =
      static_cast<vTensorImpl*>(tensor.unsafeGetTensorImpl());

  return impl->unsafe_opaque_handle();
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
