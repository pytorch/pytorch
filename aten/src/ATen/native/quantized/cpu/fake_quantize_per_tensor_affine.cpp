#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
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
    float inv_scale = 1.0f / scale;
    auto iter = TensorIterator::unary_op(Y, self);
    cpu_kernel(*iter, [&](float self) -> float {
      return (std::fmin(std::fmax(
          static_cast<int64_t>(
            std::nearbyint(self * inv_scale + zero_point)),
              quant_min), quant_max) - zero_point) * scale;
    });

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

    Tensor dX = at::zeros_like(X);
    auto iter = TensorIterator::binary_op(dX, X, dY);
    float inv_scale = 1.0f / scale;
    cpu_kernel(*iter, [&](float x, float dy) -> float {
      int64_t xq =
        static_cast<int64_t>(std::nearbyint(x * inv_scale + zero_point));
      return dy * (xq >= quant_min && xq <= quant_max);
    });
    return dX;
}
}}  // namespace at::native
