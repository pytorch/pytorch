#include <ATen/native/vulkan/ops/Factory.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

Tensor _empty_affine_quantized(
    const IntArrayRef sizes,
    const std::optional<ScalarType> dtype,
    const std::optional<c10::Layout> layout,
    const std::optional<Device> device,
    const std::optional<bool> pin_memory,
    const double scale,
    const int64_t zero_point,
    const std::optional<MemoryFormat> memory_format) {
  api::StorageType storage_type = api::StorageType::TEXTURE_3D;
  return convert_quantized(vTensor{
      api::context(),
      sizes.vec(),
      scale,
      zero_point,
      convert_dtype(dtype ? *dtype : c10::kFloat),
      storage_type,
      memory_format ? get_gpu_memory_layout(storage_type, *memory_format)
                    : api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
  });
}

static Tensor empty_memory_format(
    const IntArrayRef sizes,
    const std::optional<ScalarType> dtype,
    const std::optional<c10::Layout> layout,
    const std::optional<Device> device,
    const std::optional<bool> pin_memory,
    const std::optional<MemoryFormat> memory_format) {
  api::StorageType storage_type = api::StorageType::TEXTURE_3D;
  return convert(vTensor{
      api::context(),
      sizes.vec(),
      convert_dtype(dtype ? *dtype : c10::kFloat),
      storage_type,
      memory_format ? get_gpu_memory_layout(storage_type, *memory_format)
                    : api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
  });
}

static Tensor empty_strided(
    const IntArrayRef sizes,
    const IntArrayRef /* strides */,
    const std::optional<ScalarType> dtype,
    const std::optional<c10::Layout> layout,
    const std::optional<Device> device,
    const std::optional<bool> pin_memory) {
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
