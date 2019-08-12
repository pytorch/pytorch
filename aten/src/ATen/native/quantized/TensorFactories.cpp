#include <ATen/ATen.h>

#include <ATen/core/TensorOptions.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/core/QScheme.h>

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

} // namespace native
} // namespace at
