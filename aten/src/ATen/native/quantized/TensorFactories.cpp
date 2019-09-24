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
    const TensorOptions& options,
    double scale,
    int64_t zero_point,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(
      options.has_dtype(),
      "Must provide data type for Tensor creation functions.");
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
    IntArrayRef axis,
    const TensorOptions& options,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(
      options.has_dtype(),
      "Must provide data type for Tensor creation functions.");
  TORCH_CHECK(
      options.dtype() == kQInt8 || options.dtype() == kQUInt8,
      "Supported data type for tensor creation is int8 or uint8");
  TORCH_CHECK(scales.dim() == 1, "scale tensor must have dimension 1");
  TORCH_CHECK(
      zero_points.dim() == 1, "zero_points tensor must have dimension 1")
  TORCH_CHECK(
      scales.numel() == zero_points.numel(),
      "number of elements in scales and zero_points must match");
  TORCH_CHECK(axis.size() == 1, "only axis of size 1 is supported right now");
  double* scales_data = scales.data_ptr<double>();
  int64_t* zero_points_data = zero_points.data_ptr<int64_t>();
  std::vector<double> scale_vals(scales_data, scales_data + scales.numel());
  std::vector<int64_t> zero_point_vals(
      zero_points_data, zero_points_data + zero_points.numel());
  return new_qtensor_cpu(
      size,
      options,
      make_per_channel_affine_quantizer(
          scale_vals,
          zero_point_vals,
          axis,
          typeMetaToScalarType(options.dtype())),
      optional_memory_format.value_or(MemoryFormat::Contiguous));
}

// Provide better error message if dtype is wrong
Tensor empty_affine_quantized_other_backends_stub(
    IntArrayRef,
    const TensorOptions&,
    double,
    int64_t,
    c10::optional<c10::MemoryFormat>) {
  TORCH_CHECK(false, "Creation of quantized tensor requires quantized dtype like torch.quint8");
}

Tensor empty_per_channel_affine_quantized_other_backends_stub(
    IntArrayRef,
    const Tensor&,
    const Tensor&,
    IntArrayRef,
    const TensorOptions&,
    c10::optional<c10::MemoryFormat>) {
  TORCH_CHECK(false, "Creation of quantized tensor requires quantized dtype like torch.quint8");
}

} // namespace native
} // namespace at
