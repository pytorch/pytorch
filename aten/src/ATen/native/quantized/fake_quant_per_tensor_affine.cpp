#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/quantized/fake_quant_affine.h>

// FakeQuantize Op for PerTensorAffine quantization scheme.
namespace at {
namespace native {

// Use REGISTER_DISPATCH to run CPU and CUDA backend.
DEFINE_DISPATCH(fake_quant_tensor_stub);
DEFINE_DISPATCH(fake_quant_grad_tensor_stub);
DEFINE_DISPATCH(fake_quant_grad_learnable_sc_tensor_stub);
DEFINE_DISPATCH(fake_quant_grad_learnable_z_point_tensor_stub);

/* Fake-quantizes the 'inputs' tensor.
Args:
  X: Forward input tensor.
  dY: Backward input tensor (_backward op only).
  scale: scale of per tensor affine quantization
  zero_point: zero_point of per tensor affine quantization
  quant_min: minimum quantized value
  quant_max: maximum quantized value
Returns:
  Quantized tensor (double dtype).

*/
Tensor fake_quantize_per_tensor_affine(
    const Tensor& self,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max) {
  TORCH_CHECK(self.scalar_type() == ScalarType::Float);
  TORCH_CHECK(
      quant_min <= quant_max,
      "`quant_min` should be less than or \
        equal to `quant_max`.");
  TORCH_CHECK(
      zero_point >= quant_min && zero_point <= quant_max,
      "`zero_point` must be between `quant_min` and `quant_max`.");

  auto Y = at::empty_like(self, self.options(), MemoryFormat::Preserve);
  fake_quant_tensor_stub(
      self.device().type(), Y, self, scale, zero_point, quant_min, quant_max);
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

Tensor fake_quantize_per_tensor_affine_backward(
    const Tensor& dY,
    const Tensor& X,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max) {
  TORCH_CHECK(dY.scalar_type() == ScalarType::Float);
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
  fake_quant_grad_tensor_stub(
      X.device().type(), dX, X, dY, scale, zero_point, quant_min, quant_max);
  return dX;
}

int64_t get_zero_point_from_tensor(
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max) {
  float zero_point_fp = std::nearbyint(zero_point[0].item<float>());
  float zero_point_clamped = std::fmin(std::fmax(zero_point_fp, quant_min), quant_max);
  return static_cast<int64_t>(zero_point_clamped);
}

Tensor fake_quantize_learnable_per_tensor_affine(
    const Tensor& self,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max) {
  float scale_val = scale[0].item<float>();
  int64_t zero_point_val = native::get_zero_point_from_tensor(zero_point, quant_min, quant_max);
  return native::fake_quantize_per_tensor_affine(
    self, scale_val, zero_point_val, quant_min, quant_max);
}

std::tuple<Tensor, Tensor, Tensor> fake_quantize_learnable_per_tensor_affine_backward(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max) {
  float scale_val = scale[0].item<float>();
  int64_t zero_point_val = native::get_zero_point_from_tensor(zero_point, quant_min, quant_max);

  TORCH_CHECK(dY.scalar_type() == ScalarType::Float);
  TORCH_CHECK(X.scalar_type() == ScalarType::Float);
  TORCH_CHECK(scale.scalar_type() == ScalarType::Float);
  TORCH_CHECK(zero_point.scalar_type() == ScalarType::Float);
  TORCH_CHECK(X.numel() == dY.numel(), "`X` and `dY` are not the same size");
  TORCH_CHECK(
      quant_min <= 0 && quant_max >= 0,
      "`quant_min` should be less than or \
        equal to `quant_max`, and the quantization range should include 0.");
  TORCH_CHECK(
      zero_point_val >= quant_min && zero_point_val <= quant_max,
      "`zero_point` must be between `quant_min` and `quant_max`.");
  if (X.numel() <= 0) {
    return std::make_tuple(X, scale, zero_point);
  }

  auto dX = at::empty_like(X, X.options(), MemoryFormat::Preserve);
  fake_quant_grad_tensor_stub(
    X.device().type(), dX, X, dY, scale, zero_point, quant_min, quant_max);

  auto dScale_vec = at::empty_like(X, X.options(), MemoryFormat::Preserve);
  fake_quant_grad_learnable_sc_tensor_stub(
    scale.device().type(), dScale_vec, X, dY, scale_val, zero_point_val, quant_min, quant_max);

  auto dZeroPoint_vec = at::empty_like(X, X.options(), MemoryFormat::Preserve);
  fake_quant_grad_learnable_z_point_tensor_stub(
    zero_point.device().type(), dZeroPoint_vec, X, dY, scale_val, zero_point_val, quant_min, quant_max);

  // The total sums over the scale and zero point gradient vectors are what will be returned in the end.
  auto dScale = at::mul(dScale_vec, dX).sum().unsqueeze(0);
  auto dZeroPoint = at::mul(dZeroPoint_vec, dX).sum().unsqueeze(0);

  return std::make_tuple(dX, dScale, dZeroPoint);
}

} // namespace native
} // namespace at
