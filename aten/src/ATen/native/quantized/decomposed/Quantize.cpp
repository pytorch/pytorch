#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>

namespace at {
namespace native {
Tensor decomposed_quantize_per_tensor(
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype) {
  TORCH_CHECK(input.dtype() == at::kFloat, "Expecting input to have dtype:", at::kFloat)
  int quant_min_lower_bound = 0, quant_max_upper_bound = 0;
  if (dtype == at::kByte) {
    quant_min_lower_bound = 0;
    quant_max_upper_bound = 255;
  } else if (dtype == at::kChar) {
    quant_min_lower_bound = -128;
    quant_max_upper_bound = 127;
  } else {
    TORCH_CHECK(false, "Unsupported dtype: ", dtype);
  }

  TORCH_CHECK(
      quant_min >= quant_min_lower_bound,
      "quant_min out of bound for dtype, expected quant_min_lower_bound:",
      quant_min_lower_bound,
      " actual quant_min:",
      quant_min);

  TORCH_CHECK(
      quant_max <= quant_max_upper_bound,
      "quant_max out of bound for dtype, expected quant_max_upper_bound:",
      quant_max_upper_bound,
      " actual quant_max:",
      quant_max);

  float inv_scale = 1.0 / scale;
  return at::clamp(at::round(input * inv_scale) + zero_point, quant_min, quant_max).to(dtype);
}
}}  // namespace at::native
