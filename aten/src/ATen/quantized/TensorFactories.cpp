#include <ATen/ATen.h>
#include <c10/core/TensorOptions.h>
#include <c10/core/QScheme.h>
#include <ATen/quantized/Quantizer.h>

namespace at {
namespace native {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ empty ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// TODO: Tensor empty_quantized_cpu(IntArrayRef size, const TensorOptions& options, QScheme qscheme) {
Tensor empty_affine_quantized_cpu(IntArrayRef size, const TensorOptions& options, double scale, int64_t zero_point) {
  return new_qtensor_cpu(size, options, false, make_per_tensor_affine_quantizer(scale, zero_point));
}

}} // at::native
