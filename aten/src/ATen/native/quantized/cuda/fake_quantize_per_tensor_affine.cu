#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <cmath>
#include <ATen/native/quantized/cuda/fake_quantize_core.h>

/* FakeQuantize Op for PerTensorAffine quantization scheme */
namespace at {
namespace native {

/* Fake-quantizes the 'inputs' tensor.
Args:
  self: Forward input tensor.
  scale: scale of per tensor affine quantization
  zero_point: zero_point of per tensor affine quantization
  quant_min: minimum quantized value
  quant_max: maximum quantized value
Returns:
  Quantized tensor (double dtype).
*/
Tensor fake_quantize_per_tensor_affine_cuda(
    const Tensor& self,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max) {
  TORCH_CHECK(self.is_cuda());
  TORCH_CHECK(self.scalar_type() == ScalarType::Float);
  TORCH_CHECK(
      quant_min <= quant_max,
      "`quant_min` should be less than or \
        equal to `quant_max`.");
  TORCH_CHECK(
      zero_point >= quant_min && zero_point <= quant_max,
      "`zero_point` must be between `quant_min` and `quant_max`.");
  auto Y = at::empty_like(self, self.options(), MemoryFormat::Preserve);
  fake_quantize_slice_cuda(Y, self, scale, zero_point, quant_min, quant_max);
  return Y;
}

/* Backward path to fake-quantize the 'inputs' tensor.

Args:
  dY: Backward input tensor.
  X: Forward input tensor.
  scale: scale of per tensor affine quantization
  zero_point: zero_point of per tensor affine quantization
  quant_min: minimum quantized value
  quant_max: maximum quantized value
Returns:
  Quantized tensor (double dtype).
*/
Tensor fake_quantize_per_tensor_affine_backward_cuda(
    const Tensor& dY,
    const Tensor& X,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max) {
  TORCH_CHECK(dY.is_cuda());
  TORCH_CHECK(dY.scalar_type() == ScalarType::Float);
  TORCH_CHECK(X.is_cuda());
  TORCH_CHECK(X.scalar_type() == ScalarType::Float);
  TORCH_CHECK(X.numel() == dY.numel(), "`X` and `dY` are not the same size");
  TORCH_CHECK(
      quant_min <= quant_max,
      "`quant_min` should be less than or \
        equal to `quant_max`.");
  TORCH_CHECK(
      zero_point >= quant_min && zero_point <= quant_max,
      "`zero_point` must be between `quant_min` and `quant_max`.");
  if (X.numel() <= 0) {
    return X;
  }

  auto dX = at::empty_like(X, X.options(), MemoryFormat::Preserve);
  fake_quantize_grad_slice_cuda(
      dX, X, dY, scale, zero_point, quant_min, quant_max);
  return dX;
}

} // namespace native
} // namespace at
