#include <ATen/ATen.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/core/QScheme.h>
#include <c10/core/TensorOptions.h>

namespace at {
namespace native {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ empty ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// We explicitly pass in scale and zero_point because we don't have the infra
// ready to support quantizer in python frontend, once that is ready, we'll
// change to use quantizer
Tensor empty_affine_quantized_cpu(
    IntArrayRef size,
    c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory,
    double scale,
    int64_t zero_point,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
   /* [CHECK THIS]
  TORCH_CHECK(
      options.has_dtype(),
      "Must provide data type for Tensor creation functions.");
  */

 const auto options = TensorOptions()
        .dtype(dtype)
        .layout(layout)
        .device(device)
        .pinned_memory(pin_memory);

  return new_qtensor_cpu(
      size,
      options,
      make_per_tensor_affine_quantizer(
          scale, zero_point, typeMetaToScalarType(options.dtype())),
      optional_memory_format.value_or(MemoryFormat::Contiguous));
}

Tensor empty_per_channel_affine_quantized_cpu(
    IntArrayRef size,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis,
    c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  /* [CHECK THIS]
  TORCH_CHECK(
      options.has_dtype(),
      "Must provide data type for Tensor creation functions.");
  TORCH_CHECK(
      options.dtype() == kQInt8 || options.dtype() == kQUInt8,
      "Supported data type for tensor creation is int8 or uint8");
*/
    const auto options = TensorOptions()
        .dtype(dtype)
        .layout(layout)
        .device(device)
        .pinned_memory(pin_memory);

  return new_qtensor_cpu(
      size,
      options,
      make_per_channel_affine_quantizer(
          scales,
          zero_points,
          axis,
          typeMetaToScalarType(options.dtype())),
      optional_memory_format.value_or(MemoryFormat::Contiguous));
}

// Provide better error message if dtype is wrong
Tensor empty_affine_quantized_other_backends_stub(
    IntArrayRef,
    c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory,
    double,
    int64_t,
    c10::optional<c10::MemoryFormat>) {
  TORCH_CHECK(false, "Creation of quantized tensor requires quantized dtype like torch.quint8");
}

Tensor empty_per_channel_affine_quantized_other_backends_stub(
    IntArrayRef,
    const Tensor&,
    const Tensor&,
    int64_t,
    c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory,
    c10::optional<c10::MemoryFormat>) {
  TORCH_CHECK(false, "Creation of quantized tensor requires quantized dtype like torch.quint8");
}

} // namespace native
} // namespace at
