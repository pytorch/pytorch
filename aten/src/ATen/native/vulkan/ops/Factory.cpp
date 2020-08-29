#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

Tensor empty(
    const IntArrayRef sizes,
    const TensorOptions& options,
    const optional<MemoryFormat> memory_format) {
  TORCH_CHECK(
      !options.has_pinned_memory(),
      "'pin_memory' argument is incompatible with Vulkan tensor!");

  TORCH_CHECK(
      !options.has_memory_format() && !memory_format,
      "'memory_format' argument is incompatible with Vulkan tensor!");

  return at::detail::make_tensor<vTensorImpl>(
      DispatchKeySet(DispatchKey::Vulkan),
      options.dtype(),
      at::Device(at::kVulkan),
      vTensor(sizes),
      sizes,
      IntArrayRef{});
}

Tensor empty_strided(
    const IntArrayRef sizes,
    const IntArrayRef strides,
    const optional<ScalarType> dtype,
    const optional<Layout> layout,
    const optional<Device> device,
    const optional<bool> pin_memory) {
  return empty(
      sizes,
      TensorOptions().
        dtype(dtype).
        layout(layout).
        device(device).
        pinned_memory(pin_memory),
      c10::nullopt);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl_UNBOXED("empty.memory_format", at::native::vulkan::ops::empty);
  m.impl("empty_strided", TORCH_FN(at::native::vulkan::ops::empty_strided));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
