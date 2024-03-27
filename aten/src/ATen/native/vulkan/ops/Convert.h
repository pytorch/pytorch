#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/VulkanOpaqueTensorImpl.h>
#include <ATen/native/vulkan/api/Tensor.h>
#include <c10/util/accumulate.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

using vTensorImpl = VulkanOpaqueTensorImpl<vTensor>;

inline Tensor convert(const vTensor& tensor) {
  return at::detail::make_tensor<vTensorImpl>(
      DispatchKeySet(DispatchKey::Vulkan),
      c10::scalarTypeToTypeMeta(tensor.dtype()),
      at::Device(at::kVulkan),
      tensor,
      tensor.sizes(),
      tensor.strides());
}

inline Tensor convert_quantized(const vTensor& tensor) {
  TORCH_CHECK(tensor.is_quantized(), "Not a Quantized Tensor");
  return at::detail::make_tensor<vTensorImpl>(
      DispatchKeySet(DispatchKey::Vulkan),
      c10::scalarTypeToTypeMeta(tensor.dtype()),
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
