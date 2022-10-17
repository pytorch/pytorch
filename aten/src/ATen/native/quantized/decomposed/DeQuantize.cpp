#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>

namespace at {
namespace native {
Tensor decomposed_dequantize_per_tensor(const Tensor& input, double scale, int64_t zero_point, ScalarType dtype) {
  TORCH_CHECK(input.dtype() == dtype, "Expecting input to have dtype:", dtype)
  if (dtype == at::kByte || dtype == at::kChar) {
    return (input.to(at::kFloat) - zero_point) * scale;
  } else {
    TORCH_CHECK(false, "Unsupported dtype: ", dtype);
  }
}
}}  // namespace at::native
