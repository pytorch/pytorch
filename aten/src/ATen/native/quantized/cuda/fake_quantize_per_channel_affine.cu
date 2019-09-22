#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <cmath>

/* FakeQuantize Op for PerChannelAffine quantization scheme */
namespace at {
namespace native {

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
Tensor fake_quantize_per_channel_affine_cuda(
    const Tensor& self,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max) {


  TORCH_CHECK(self.is_cuda());
  TORCH_CHECK(self.scalar_type() == ScalarType::Float);
  TORCH_CHECK(
      quant_min <= quant_max,
      "`quant_min` should be less than or \
        equal to `quant_max`.");
/*  TORCH_CHECK(
      zero_point >= quant_min && zero_point <= quant_max,
      "`zero_point` must be between `quant_min` and `quant_max`.");
      */
  auto Y = at::empty_like(self);
  for (int i = 0; i < self.size(0); i++)
  {
    auto Z = self.slice(0,i,i+1);
    auto Z1 = Y.slice(0,i,i+1);
    float sc = scale[i].item().toFloat();
    int64_t zp = zero_point[i].item().toLong();
    float inv_scale = 1.0f/sc;
    at::cuda::CUDA_tensor_apply2<float, float>(
      Z, Z1, [=] __device__(const float& input_val, float& result_val) {
        result_val = (fminf(
                          quant_max,
                          fmaxf(
                              quant_min,
                              static_cast<int64_t>(std::nearbyint(
                                  input_val * inv_scale + zp)))) -
                      zp) *
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
  - quantization range [quant_min, quant_max]
*/
Tensor fake_quantize_per_channel_affine_backward_cuda(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& scale,
    const Tensor& zero_point,
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
/*  TORCH_CHECK(
      zero_point >= quant_min && zero_point <= quant_max,
      "`zero_point` must be between `quant_min` and `quant_max`.");
*/
  if (X.numel() <= 0) {
    return X;
  }


  auto dX = dY.clone();
  for (int i = 0; i < X.size(0); i++)
  {
    auto dY_slice = dY.slice(0,i,i+1);
    auto X_slice = X.slice(0,i,i+1);
    auto dX_slice = dX.slice(0,i,i+1);
    float sc = scale[i].item().toFloat();
    int64_t zp = scale[i].item().toLong();
    float inv_scale = 1.0f/sc;


  at::cuda::CUDA_tensor_apply3<float, float, float>(
      dY_slice, X_slice, dX_slice, [=] __device__(const float& dy, const float& x, float& dx) {
        int64_t Xq = std::nearbyint(x * inv_scale + zp);
        dx = (Xq >= quant_min && Xq <= quant_max) * dy;
      });
    }
  return dX;
}

} // namespace native
} // namespace at
