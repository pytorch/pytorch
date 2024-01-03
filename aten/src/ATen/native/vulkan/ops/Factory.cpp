#include <ATen/native/vulkan/ops/Factory.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

Tensor _empty_affine_quantized(
    const IntArrayRef sizes,
    const c10::optional<ScalarType> dtype,
    const c10::optional<c10::Layout> layout,
    const c10::optional<Device> device,
    const c10::optional<bool> pin_memory,
    const double scale,
    const int64_t zero_point,
    const optional<MemoryFormat> memory_format) {
  return convert_quantized(vTensor{
      api::context(),
      sizes,
      scale,
      zero_point,
      dtype ? *dtype : c10::kFloat,
      api::StorageType::TEXTURE_3D,
      memory_format ? *memory_format : c10::MemoryFormat::Contiguous,
  });
}

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
      dtype ? *dtype : c10::kFloat,
      api::StorageType::TEXTURE_3D,
      memory_format ? *memory_format : c10::MemoryFormat::Contiguous,
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
      sizes, dtype, layout, device, pin_memory, c10::MemoryFormat::Contiguous);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::empty.memory_format"),
      at::native::vulkan::ops::empty_memory_format);
  m.impl(
      TORCH_SELECTIVE_NAME("aten::_empty_affine_quantized"),
      at::native::vulkan::ops::_empty_affine_quantized);
  m.impl(
      TORCH_SELECTIVE_NAME("aten::empty_strided"),
      TORCH_FN(at::native::vulkan::ops::empty_strided));
}

#endif /* USE_VULKAN_API */

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
