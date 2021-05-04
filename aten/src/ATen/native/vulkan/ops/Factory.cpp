#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

Tensor empty_memory_format(
    const IntArrayRef sizes,
    const c10::optional<ScalarType> dtype,
    const c10::optional<c10::Layout> layout,
    const c10::optional<Device> device,
    const c10::optional<bool> pin_memory,
    const optional<MemoryFormat> memory_format) {
  return convert(vTensor{
      api::context(),
      sizes,
      TensorOptions()
          .dtype(dtype)
          .layout(layout)
          .device(device)
          .pinned_memory(pin_memory)
          .memory_format(memory_format),
    });
}

Tensor empty_strided(
    const IntArrayRef sizes,
    const IntArrayRef /* strides */,
    const optional<ScalarType> dtype,
    const optional<c10::Layout> layout,
    const optional<Device> device,
    const optional<bool> pin_memory) {
  return empty_memory_format(
      sizes,
      dtype,
      layout,
      device,
      pin_memory,
      c10::MemoryFormat::Contiguous);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl("empty.memory_format", at::native::vulkan::ops::empty_memory_format);
  m.impl("empty_strided", TORCH_FN(at::native::vulkan::ops::empty_strided));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
