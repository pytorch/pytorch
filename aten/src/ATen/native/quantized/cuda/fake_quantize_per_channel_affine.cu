#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <cmath>
#include "fake_quantize_core.h"

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
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max) {


  TORCH_CHECK(self.is_cuda());
  TORCH_CHECK(self.scalar_type() == ScalarType::Float);

  TORCH_CHECK(scale.size(0) == zero_point.size(0),
  "scale and zero-point need to have the same dimensions");
  TORCH_CHECK(scale.size(0) == self.size(axis),
  "dimensions of scale and zero-point are not consistent with input tensor")


  TORCH_CHECK(
      quant_min <= quant_max,
      "`quant_min` should be less than or \
        equal to `quant_max`.");


  TORCH_CHECK(at::min(zero_point).item().toLong() >= quant_min &&
              at::max(zero_point).item().toLong() <= quant_max,
      "`zero_point` must be between `quant_min` and `quant_max`.");

  TORCH_CHECK(axis >= 0 &&
              axis <= self.dim(),
      "`axis` must be between 0 and number of dimensions of input");

  auto Y = at::empty_like(self);
  for (int i = 0; i < self.size(axis); i++)
  {
    auto X_slice = self.slice(axis,i,i+1);
    auto Y_slice = Y.slice(axis,i,i+1);
    float sc = scale[i].item().toFloat();
    int64_t zp = zero_point[i].item().toLong();
    fake_quantize_slice_cuda(Y_slice, X_slice, sc, zp, quant_min, quant_max);
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
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max) {

  TORCH_CHECK(dY.is_cuda());

  TORCH_CHECK(dY.scalar_type() == ScalarType::Float);
  TORCH_CHECK(X.scalar_type() == ScalarType::Float);

  TORCH_CHECK(X.numel() == dY.numel(), "`X` and `dY` are not the same size");
  TORCH_CHECK(
      quant_min <= quant_max,
      "`quant_min` should be less than or \
        equal to `quant_max`.");

  TORCH_CHECK(scale.size(0) == zero_point.size(0),
  "scale and zero-point need to have the same dimensions")
  TORCH_CHECK(scale.size(0) == X.size(axis),
  "dimensions of scale and zero-point are not consistent with input tensor")


  TORCH_CHECK(
      quant_min <= quant_max,
      "`quant_min` should be less than or \
        equal to `quant_max`.");


  TORCH_CHECK(at::min(zero_point).item().toLong() >= quant_min &&
              at::max(zero_point).item().toLong() <= quant_max,
      "`zero_point` must be between `quant_min` and `quant_max`.");

  TORCH_CHECK(axis >= 0 &&
              axis <= X.dim(),
      "`axis` must be between 0 and number of dimensions of input");


  if (X.numel() <= 0) {
    return X;
  }


  auto dX = dY.clone();
  for (int i = 0; i < X.size(axis); i++)
  {
    auto dY_slice = dY.slice(axis,i,i+1);
    auto X_slice = X.slice(axis,i,i+1);
    auto dX_slice = dX.slice(axis,i,i+1);
    float sc = scale[i].item().toFloat();
    int64_t zp = scale[i].item().toLong();
    fake_quantize_grad_slice_cuda(dX_slice, X_slice, dY_slice, sc, zp, quant_min, quant_max);
  }
  return dX;
}

} // namespace native
} // namespace at
