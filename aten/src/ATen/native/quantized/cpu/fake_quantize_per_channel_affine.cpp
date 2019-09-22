#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

/* FakeQuantize Op for PerChannelAffine quantization scheme */
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
Tensor fake_quantize_per_channel_affine_cpu(
    const Tensor& self,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max) {
  TORCH_CHECK(self.scalar_type() == ScalarType::Float);
  TORCH_CHECK(
      quant_min <= quant_max,
      "`quant_min` should be less than or \
        equal to `quant_max`.");

//      return self;
/*  TORCH_CHECK(
#      zero_point >= quant_min && zero_point <= quant_max,
#      "`zero_point` must be between `quant_min` and `quant_max`.");
*/
  auto Y = at::empty_like(self);
  for (int i = 0; i < self.size(0); i++)
  {
    auto input_slice = self.slice(0,i,i+1);
    auto output_slice = Y.slice(0,i,i+1);

    float sc = scale[i].item().toFloat();
    float inv_scale = 1.0f / sc;
    int64_t z_point = zero_point[i].item().toLong();
    auto iter = TensorIterator::unary_op(output_slice, input_slice);
    cpu_kernel(iter, [&](float self) -> float {
      return (std::fmin(
                  std::fmax(
                      static_cast<int64_t>(
                          std::nearbyint(self * inv_scale + z_point)),
                      quant_min),
                  quant_max) -
              z_point) *
          sc;
    });

  }

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
Tensor fake_quantize_per_channel_affine_backward_cpu(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max) {

  /*
  TORCH_CHECK(dY.scalar_type() == ScalarType::Float);
  TORCH_CHECK(X.scalar_type() == ScalarType::Float);
  */

  TORCH_CHECK(X.numel() == dY.numel(), "`X` and `dY` are not the same size");
  TORCH_CHECK(
      quant_min <= quant_max,
      "`quant_min` should be less than or \
        equal to `quant_max`.");
  /*TORCH_CHECK(
      zero_point >= quant_min && zero_point <= quant_max,
      "`zero_point` must be between `quant_min` and `quant_max`.");
      */

  if (X.numel() <= 0) {
    return X;
  }

  Tensor dX = at::zeros_like(X);

  for (int i = 0; i < X.size(0); i++)
  {
    auto ZX = X.slice(0,i,i+1);
    auto ZdY = dY.slice(0,i,i+1);
    auto ZdX = dX.slice(0,i,i+1);

    float sc = scale[i].item().toFloat();
    float inv_scale = 1.0f / sc;
    int64_t z_point = zero_point[i].item().toLong();
    auto iter = TensorIterator::binary_op(ZdX, ZX, ZdY);
    cpu_kernel(iter, [&](float x, float dy) -> float {
    int64_t xq =
        static_cast<int64_t>(std::nearbyint(x * inv_scale + z_point));
    return dy * (xq >= quant_min && xq <= quant_max);
  });
}
  return dX;

}
} // namespace native
} // namespace at
