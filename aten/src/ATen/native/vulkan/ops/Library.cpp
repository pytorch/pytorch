#include <ATen/native/vulkan/ops/Library.h>
#include <ATen/native/vulkan/VulkanOpaqueTensorImpl.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

using VulkanTensorImpl = VulkanOpaqueTensorImpl<vTensor>;

Tensor empty(
    const IntArrayRef sizes,
    const optional<ScalarType> dtype,
    const optional<Layout> layout,
    const optional<Device> device,
    const optional<bool> pin_memory,
    const optional<MemoryFormat> memory_format) {
  return at::detail::make_tensor<VulkanTensorImpl>(
      DispatchKeySet(DispatchKey::Vulkan),
      dtype,
      at::Device(at::kVulkan),
      vTensor(sizes),
      sizes,
      IntArrayRef{});
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl("empty.memory_format", TORCH_FN(at::native::vulkan::api::empty));
}

#endif /* USE_VULKAN_API */

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
