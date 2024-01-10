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

/**
 * Determines an appropriate GPU Memory Layout qualifier based on the the
 * StorageType requested and the c10::MemoryFormat specified.
 */
inline api::GPUMemoryLayout get_gpu_memory_layout(
    const api::StorageType storage_type,
    const c10::MemoryFormat memory_format) {
  if (storage_type == api::StorageType::BUFFER) {
    switch (memory_format) {
      case c10::MemoryFormat::Contiguous:
        return api::GPUMemoryLayout::TENSOR_WIDTH_PACKED;
      case c10::MemoryFormat::ChannelsLast:
        return api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED;
      default:
        VK_THROW("Invalid memory format used to create vTensor!");
    }
  }
  // For texture storage, always return a memory layout that packs the channels
  // dimension. for now. With the way texture storage currently works, for 2-dim
  // tensors, a channel dimension is added, as well as 3 channels of zero
  // padding resulting in a final shape of {4, H, W}. For 1-dim tensors, it is
  // unsqueezed to size {1, 1, L} and 3 channels of zero padding are added to
  // produce a final size of {4, 1, L}. This is to ensure that physical texture
  // positions correspond directly to logical tensor coordinates (so
  // texelFetch(ivec3(x, y, 0), 0) will correspond to tensor[y, x].
  //
  // TODO(ssjia): have 2D and 1D tensors use TENSOR_WIDTH_PACKED by default.
  return api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED;
}

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
