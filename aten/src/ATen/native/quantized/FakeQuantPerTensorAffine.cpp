#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/quantized/FakeQuantAffine.h>

// FakeQuantize Op for PerTensorAffine quantization scheme.
namespace at {
namespace native {

// Use REGISTER_DISPATCH to run CPU and CUDA backend.
DEFINE_DISPATCH(fake_quant_tensor_cachemask_stub);
DEFINE_DISPATCH(fake_quant_grad_learnable_tensor_stub);
DEFINE_DISPATCH(fake_quant_tensor_cachemask_tensor_qparams_stub);

/* Fake-quantizes the 'inputs' tensor.

Args:
  self: Forward input tensor.
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
  const auto res = at::fake_quantize_per_tensor_affine_cachemask(
      self, scale, zero_point, quant_min, quant_max);
  return std::get<0>(res);
}

Tensor fake_quantize_per_tensor_affine(
    const Tensor& self,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max) {
  const auto res = at::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
      self, scale, zero_point, at::ones(1, self.options().dtype(at::kLong)), quant_min, quant_max);
  return std::get<0>(res);
}

/* Fake-quantizes the 'inputs' tensor, saving a mask for the backward pass.

This is numerically equivalent to `fake_quantize_per_tensor_affine`,
but has a lower memory overhead in the backward pass.

Args:
  self: Forward input tensor.
  scale: scale of per tensor affine quantization
  zero_point: zero_point of per tensor affine quantization
  quant_min: minimum quantized value
  quant_max: maximum quantized value

Returns:
  Quantized tensor (double dtype).
  Mask (bool dtype).
*/
std::tuple<Tensor, Tensor> fake_quantize_per_tensor_affine_cachemask(
    const Tensor& self,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max) {
  TORCH_CHECK(
      quant_min <= quant_max,
      "`quant_min` should be less than or \
        equal to `quant_max`.");
  TORCH_CHECK(
      zero_point >= quant_min && zero_point <= quant_max,
      "`zero_point` must be between `quant_min` and `quant_max`.");

  auto Y = at::empty_like(self, self.options(), MemoryFormat::Preserve);
  auto mask = at::empty_like(self, at::kBool, MemoryFormat::Preserve);
  fake_quant_tensor_cachemask_stub(
      self.device().type(), Y, mask, self, scale, zero_point, quant_min, quant_max);
  // TODO(future, optional): look into packing the mask further (BoolTensor uses
  //   1 byte per element, we only need 1 bit per element).
  return std::make_tuple(Y, mask);
}

std::tuple<Tensor, Tensor> _fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
    const Tensor& self,
    const Tensor& scale,
    const Tensor& zero_point,
    const Tensor& fake_quant_enabled,
    int64_t quant_min,
    int64_t quant_max) {
  TORCH_CHECK(
      quant_min <= quant_max,
      "`quant_min` should be less than or \
        equal to `quant_max`.");
  auto Y = at::empty_like(self, self.options(), MemoryFormat::Preserve);
  auto mask = at::empty_like(self, at::kBool, MemoryFormat::Preserve);
  fake_quant_tensor_cachemask_tensor_qparams_stub(
      self.device().type(), Y, mask, self, scale, zero_point, fake_quant_enabled, quant_min, quant_max);
  // TODO(future, optional): look into packing the mask further (BoolTensor uses
  //   1 byte per element, we only need 1 bit per element).
  return std::make_tuple(Y, mask);
}

/* Backward path to fake-quantize the 'inputs' tensor, with mask.

Args:
  dY: output grad.
  mask: mask tensor from the forward pass.

Returns:
  dX (input grad).
*/
Tensor fake_quantize_per_tensor_affine_cachemask_backward(
    const Tensor& dY,
    const Tensor& mask) {
  TORCH_CHECK(mask.scalar_type() == ScalarType::Bool);
  TORCH_CHECK(mask.sym_numel() == dY.sym_numel(),
      "`mask` and `dY` are not the same size: ",
      "`mask` is size ", mask.sym_numel(), " and `dY` is size ", dY.sym_numel());
  if (dY.sym_numel() <= 0) {
    return dY;
  }
  // Note: no additional kernels needed, since mask is pre-computed
  // and we can use the existing tensor multiplication kernels.
  return dY * mask;
}

static int64_t _get_zero_point_from_tensor(
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max,
    bool is_forward) {
  float zero_point_fp = zero_point[0].item<float>();
  zero_point_fp = is_forward ? std::nearbyint(zero_point_fp) : zero_point_fp + 0.5f;
  float zero_point_clamped = std::min(std::max(zero_point_fp, static_cast<float>(quant_min)),
                                       static_cast<float>(quant_max));
  return static_cast<int64_t>(zero_point_clamped);
}

Tensor _fake_quantize_learnable_per_tensor_affine(
    const Tensor& self,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max,
    double grad_factor) {
  float scale_val = scale[0].item<float>();
  int64_t zero_point_val = native::_get_zero_point_from_tensor(zero_point, quant_min, quant_max, true);
  return native::fake_quantize_per_tensor_affine(
    self, scale_val, zero_point_val, quant_min, quant_max);
}

std::tuple<Tensor, Tensor, Tensor> _fake_quantize_learnable_per_tensor_affine_backward(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max,
    double grad_factor) {
  /* The gradients for scale and zero point are calculated as below:
     Let Xfq be the fake quantized version of X.
     Let Xq be the quantized version of X (clamped at qmin and qmax).
     Let Delta and z be the scale and the zero point.
     :math:
      \frac{d\Delta }{dx} =
        \begin{cases}
          q_{\min} - z& \text{ if } X_q= q_{\min} \\
          q_{\max} - z& \text{ if } X_q= q_{\max} \\
          (X_{fq} - X) / \Delta & \text{ else }
        \end{cases}

      \frac{dz }{dx} =
        \begin{cases}
          -\Delta& \text{ if } X_q= q_{\min} \text{ or } X_q = q_{\max} \\
          0 & \text{ else }
        \end{cases}
  */
  float scale_val = scale[0].item<float>();
  float inv_scale_val = 1.0f / scale_val;
  int64_t zero_point_val = native::_get_zero_point_from_tensor(zero_point, quant_min, quant_max, false);

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
  auto dScale_vec = at::empty_like(X, X.options(), MemoryFormat::Preserve);
  auto dZeroPoint_vec = at::empty_like(X, X.options(), MemoryFormat::Preserve);

  auto iter = TensorIteratorConfig()
    .add_output(dX)
    .add_output(dScale_vec)
    .add_output(dZeroPoint_vec)
    .add_input(X)
    .add_input(dY)
    .build();

  fake_quant_grad_learnable_tensor_stub(
    X.device().type(), iter, scale_val, inv_scale_val, zero_point_val, quant_min, quant_max, grad_factor);

  // The total sums over the scale and zero point gradient vectors are what will be returned in the end.
  auto dScale = dScale_vec.sum().unsqueeze(0).to(scale.device());
  auto dZeroPoint = dZeroPoint_vec.sum().unsqueeze(0).to(zero_point.device());

  return std::make_tuple(dX, dScale, dZeroPoint);
}

} // namespace native
} // namespace at
