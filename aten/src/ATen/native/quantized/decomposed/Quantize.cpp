#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>

namespace at {
namespace native {
Tensor decomposed_quantize_per_tensor(const Tensor& input, double scale, int64_t zero_point, ScalarType dtype) {
  TORCH_CHECK(input.dtype() == at::kFloat, "Expecting input to have dtype:", at::kFloat)
  int qmin = 0, qmax = 0;
  if (dtype == at::kByte) {
    qmin = 0;
    qmax = 255;
  } else if (dtype == at::kChar) {
    qmin = -128;
    qmax = 127;
  } else {
    TORCH_CHECK(false, "Unsupported dtype: ", dtype);
  }
  float inv_scale = 1.0 / scale;
  return at::clamp(at::round(input * inv_scale) + zero_point, qmin, qmax).to(dtype);
}
}}  // namespace at::native
