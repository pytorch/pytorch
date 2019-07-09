#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

/* FakeQuantize Op for PerTensorAffine quantization scheme */
namespace at {
namespace native {

/* Fake-quantizes the 'inputs' tensor.
Args:
  X: Forward input tensor.
  dY: Backward input tensor (_backward op only).
  scale: scale of per tensor affine quantization
  zero_point: zero_point of per tensor affine quantization
  quant_min: minimum quantized value
  quant_max: maximum quantized value
  quant_delay: Count of global steps for which to delay the quantization.
               See note below.
  iter: The current quantization iteration used for `quant_delay`.
Returns:
  Quantized tensor (double dtype).

Notes:
  - quant_delay might be set to non-zero to help weights stabilize in the
    beginning of the training.
  - quantization range [quant_min, quant_max]
*/
Tensor fake_quantize_per_tensor_affine_cpu(
      const Tensor& self,
      double scale,
      int64_t zero_point,
      int64_t quant_min,
      int64_t quant_max
    ) {
    TORCH_CHECK(self.scalar_type() == ScalarType::Float);
    TORCH_CHECK(quant_min <= quant_max, "`quant_min` should be less than or \
        equal to `quant_max`.");
    TORCH_CHECK(zero_point >= quant_min && zero_point <= quant_max,
        "`zero_point` must be between `quant_min` and `quant_max`.");

    auto Y = at::empty_like(self);

    double inv_scale = 1.0f / scale;
    Y = ((self * inv_scale + zero_point).round()
      .clamp_min(quant_min).clamp_max(quant_max) - zero_point) * scale;
    return Y;
}

/* Backward path to fake-quantize the 'inputs' tensor.

Args:
  X: Forward input tensor.
  dY: Backward input tensor.
  scale: scale of per tensor affine quantization
  zero_point: zero_point of per tensor affine quantization
  quant_min: minimum quantized value
  quant_max: maximum quantized value
  quant_delay: Count of global steps for which to delay the quantization.
               See note in forward.
  iter: The current quantization iteration used for `quant_delay`.
Returns:
  Quantized tensor (double dtype).

Notes:
  - quant_delay might be set to non-zero to help weights stabilize in the
    beginning of the training.
  - quantization range [0, 2^bits - 1]
*/
Tensor fake_quantize_per_tensor_affine_backward_cpu(
      const Tensor& dY,
      const Tensor& X,
      double scale,
      int64_t zero_point,
      int64_t quant_min,
      int64_t quant_max) {
    TORCH_CHECK(dY.scalar_type() == ScalarType::Float);
    TORCH_CHECK(X.scalar_type() == ScalarType::Float);
    TORCH_CHECK(X.numel() == dY.numel(), "`X` and `dY` are not the same size");
    TORCH_CHECK(quant_min <= quant_max, "`quant_min` should be less than or \
        equal to `quant_max`.");
    TORCH_CHECK(zero_point >= quant_min && zero_point <= quant_max,
        "`zero_point` must be between `quant_min` and `quant_max`.");
    if (X.numel() <= 0) {
      return X;
    }

    double inv_scale = 1.0f / scale;
    Tensor Xq = (X * inv_scale + zero_point).round();
    Tensor mask_min = (Xq >= quant_min);
    Tensor mask_max = (Xq <= quant_max);
    Tensor mask = mask_min * mask_max;
    Tensor dX = mask.type_as(dY) * dY;
    return dX;
}
}}  // namespace at::native
