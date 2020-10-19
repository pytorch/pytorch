#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

Tensor empty_memory_format(
    const IntArrayRef sizes,
    const TensorOptions& options_,
    const optional<MemoryFormat> memory_format = c10::nullopt) {
  TORCH_CHECK(
      !(options_.has_memory_format() && memory_format.has_value()),
      "Cannot set memory_format both in TensorOptions and explicit argument!");

  const TensorOptions options = options_.merge_in(
      TensorOptions().memory_format(memory_format));
  verify(options);

  return convert(vTensor{
      api::context(),
      sizes,
      options,
    });
}

Tensor empty_strided(
    const IntArrayRef sizes,
    const IntArrayRef /* strides */,
    const optional<ScalarType> dtype,
    const optional<Layout> layout,
    const optional<Device> device,
    const optional<bool> pin_memory) {
  return empty_memory_format(
      sizes,
      TensorOptions().
          dtype(dtype).
          layout(layout).
          device(device).
          pinned_memory(pin_memory));
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl_UNBOXED("empty.memory_format", at::native::vulkan::ops::empty_memory_format);
  m.impl("empty_strided", TORCH_FN(at::native::vulkan::ops::empty_strided));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
