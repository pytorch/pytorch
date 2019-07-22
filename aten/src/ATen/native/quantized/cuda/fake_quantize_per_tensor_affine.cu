#include <ATen/ATen.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/NativeFunctions.h>
#include <cmath>

/* FakeQuantize Op for PerTensorAffine quantization scheme */
namespace at {
namespace native{

/* Fake-quantizes the 'inputs' tensor.
Args:
  X: Forward input tensor.
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
Tensor fake_quantize_per_tensor_affine_cuda(
      const Tensor& self,
      double scale,
      int64_t zero_point,
      int64_t quant_min,
      int64_t quant_max
    ) {
    TORCH_CHECK(self.is_cuda());
    TORCH_CHECK(self.scalar_type() == ScalarType::Float);
    TORCH_CHECK(quant_min <= quant_max, "`quant_min` should be less than or \
        equal to `quant_max`.");
    TORCH_CHECK(zero_point >= quant_min && zero_point <= quant_max,
        "`zero_point` must be between `quant_min` and `quant_max`.");
    auto Y = at::empty_like(self);

    float inv_scale = 1.0f / scale;
    at::cuda::CUDA_tensor_apply2<float, float>(
        self,
        Y,
        [=] __device__ (
            const float& input_val,
            float& result_val) {
          result_val =
            (fminf(quant_max, fmaxf(quant_min,
              static_cast<int64_t>(
                std::nearbyint(input_val * inv_scale + zero_point))))
                - zero_point) * scale;
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
  - quantization range [quant_min, quant_max]
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
    TORCH_CHECK(quant_min <= quant_max, "`quant_min` should be less than or \
        equal to `quant_max`.");
    TORCH_CHECK(zero_point >= quant_min && zero_point <= quant_max,
        "`zero_point` must be between `quant_min` and `quant_max`.");
    if (X.numel() <= 0) {
      return X;
    }

    auto dX = dY.clone();

    float inv_scale = 1.0f / scale;
    at::cuda::CUDA_tensor_apply3<float, float, float>(
        dY,
        X,
        dX,
        [=] __device__ (
            const float& dy,
            const float& x,
            float& dx) {
          int64_t Xq = std::nearbyint(x * inv_scale + zero_point);
          dx = (Xq >= quant_min && Xq <= quant_max) *  dy;
        });
    return dX;
}

}} // namespace at::native
