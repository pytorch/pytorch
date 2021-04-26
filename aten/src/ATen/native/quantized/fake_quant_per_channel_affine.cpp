#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/quantized/fake_quant_affine.h>

#include <c10/util/irange.h>

// FakeQuantize Op for PerChannelAffine quantization scheme.
namespace at {
namespace native {

// Use REGISTER_DISPATCH to run CPU and CUDA backend.
DEFINE_DISPATCH(fake_quant_per_channel_cachemask_stub);
DEFINE_DISPATCH(fake_quant_grad_learnable_channel_stub);

/* Per channel fake-quantizes the 'inputs' tensor.
Args:
  X: Forward input tensor.
  dY: Backward input tensor (_backward op only).
  scale: scale of per channel affine quantization
  zero_point: zero_point of per channel affine quantization
  axis: int specifying the axis to be quantized
  quant_min: minimum quantized value
  quant_max: maximum quantized value
Returns:
  Fake quantized tensor (double dtype).

*/

Tensor fake_quantize_per_channel_affine(
    const Tensor& self,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max) {
  const auto res = at::fake_quantize_per_channel_affine_cachemask(
      self, scale, zero_point, axis, quant_min, quant_max);
  return std::get<0>(res);
}

std::tuple<Tensor, Tensor> fake_quantize_per_channel_affine_cachemask(
    const Tensor& self,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max) {
  TORCH_CHECK(self.scalar_type() == ScalarType::Float);
  TORCH_CHECK(scale.scalar_type() == ScalarType::Float,
              "Scale must be Float, found ", scale.scalar_type());
  TORCH_CHECK(zero_point.scalar_type() == ScalarType::Long,
              "Zero-point must be Long, found ", zero_point.scalar_type());
  TORCH_CHECK(scale.dim() == 1, "scale should be a 1-D tensor");
  TORCH_CHECK(zero_point.dim() == 1, "zero point should be a 1-D tensor");
  TORCH_CHECK(
      scale.numel() == zero_point.numel(),
      "scale and zero-point need to have the same dimensions");
  TORCH_CHECK(
      scale.numel() == self.size(axis),
      "dimensions of scale and zero-point are not consistent with input tensor")

  TORCH_CHECK(
      quant_min <= quant_max,
      "`quant_min` should be less than or \
        equal to `quant_max`.");

  TORCH_CHECK(
      at::min(zero_point).item().toLong() >= quant_min &&
          at::max(zero_point).item().toLong() <= quant_max,
      "`zero_point` must be between `quant_min` and `quant_max`.");

  TORCH_CHECK(
      axis >= 0 && axis <= self.dim(),
      "`axis` must be between 0 and number of dimensions of input");

  auto Y = at::empty_like(self, self.options(), MemoryFormat::Preserve);
  auto mask = at::empty_like(self, at::kBool, MemoryFormat::Preserve);

  std::vector<int64_t> expected_shape(self.dim(), 1);
  expected_shape[axis] = self.size(axis);

  TensorIterator iter = TensorIteratorConfig()
    .check_all_same_dtype(false)
    .add_output(Y)
    .add_input(self)
    .add_input(native::_unsafe_view(scale, expected_shape))
    .add_input(native::_unsafe_view(zero_point, expected_shape))
    .build();

  // TODO(future, optional): read once, write twice.  Not done at the moment
  //   for simplicity, as we do not expect this to be a bottleneck.
  TensorIterator iter_mask = TensorIteratorConfig()
    .check_all_same_dtype(false)
    .add_output(mask)
    .add_input(self)
    .add_input(native::_unsafe_view(scale, expected_shape))
    .add_input(native::_unsafe_view(zero_point, expected_shape))
    .build();

  // TODO(future, optional): look into packing the mask further (BoolTensor uses
  //   1 byte per element, we only need 1 bit per element).
  fake_quant_per_channel_cachemask_stub(iter.device_type(), iter, iter_mask, quant_min, quant_max);
  return std::make_tuple(Y, mask);
}

/* Backward path to fake-quantize the 'inputs' tensor per channel, with mask.

Args:
  dY: output grad.
  mask: mask tensor from the forward pass.

Returns:
  dX (input grad).
*/
Tensor fake_quantize_per_channel_affine_cachemask_backward(
    const Tensor& dY,
    const Tensor& mask) {
  TORCH_CHECK(dY.scalar_type() == ScalarType::Float);
  TORCH_CHECK(mask.scalar_type() == ScalarType::Bool);
  TORCH_CHECK(mask.numel() == dY.numel(),
      "`mask` and `dY` are not the same size: ",
      "`mask` is size ", mask.numel(), " and `dY` is size ", dY.numel());
  if (dY.numel() <= 0) {
    return dY;
  }
  // Note: no additional kernels needed, since mask is pre-computed
  // and we can use the existing tensor multiplication kernels.
  return dY * mask;
}

Tensor _get_rounded_zero_point(
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max) {
  // This assumes the per channel zero point vector is single-dimensioned.
  return zero_point.round().clamp_(quant_min, quant_max);
}

Tensor _fake_quantize_learnable_per_channel_affine(
    const Tensor& self,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max,
    double grad_factor) {
  Tensor zero_point_rounded = _get_rounded_zero_point(zero_point, quant_min, quant_max).to(at::kLong);
  return native::fake_quantize_per_channel_affine(
    self, scale, zero_point_rounded, axis, quant_min, quant_max);
}

std::tuple<Tensor, Tensor, Tensor> _fake_quantize_learnable_per_channel_affine_backward(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t axis,
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
  auto zero_point_rounded = _get_rounded_zero_point(zero_point, quant_min, quant_max);

  TORCH_CHECK(dY.scalar_type() == ScalarType::Float);
  TORCH_CHECK(X.scalar_type() == ScalarType::Float);
  TORCH_CHECK(scale.scalar_type() == ScalarType::Float);
  TORCH_CHECK(zero_point.scalar_type() == ScalarType::Float);

  TORCH_CHECK(X.sizes() == dY.sizes(), "`X` and `dY` are not the same size");
  TORCH_CHECK(
      quant_min <= 0 && quant_max >= 0,
      "Expecting `quant_min` <= 0 and `quant_max` >= 0");
  TORCH_CHECK(scale.dim() == 1, "scale should be a 1-D tensor");
  TORCH_CHECK(zero_point.dim() == 1, "zero point should be a 1-D tensor");
  TORCH_CHECK(
      scale.numel() == zero_point.numel(),
      "scale and zero-point need to have the same dimensions");
  TORCH_CHECK(
      scale.numel() == X.size(axis),
      "dimensions of scale and zero-point are not consistent with input tensor")

  TORCH_CHECK(
      at::min(zero_point_rounded).item().toLong() >= quant_min &&
          at::max(zero_point_rounded).item().toLong() <= quant_max,
      "`zero_point` must be between `quant_min` and `quant_max`.");

  TORCH_CHECK(
      axis >= 0 && axis < X.dim(),
      "`axis` must be between 0 and number of dimensions of input");

  if (X.numel() <= 0) {
    return std::make_tuple(X, scale, zero_point);
  }

  auto dX = at::empty_like(X, X.options(), MemoryFormat::Preserve);
  auto dScale_vec = at::empty_like(X, X.options(), MemoryFormat::Preserve);
  auto dZeroPoint_vec = at::empty_like(X, X.options(), MemoryFormat::Preserve);
  int numDimensions = X.ndimension();

  // Create an axis mask for vectorizing and reshaping the scale and zero point tensors
  // into the same shapes as X along the channel axis.
  int64_t* axis_mask = (int64_t *) calloc(numDimensions, sizeof(int64_t));
  for (int i = 0; i < numDimensions; ++i) {
    axis_mask[i] = (i == axis) ? X.size(axis) : 1;
  }
  auto X_shape = X.sizes();
  auto scale_vectorized = scale.reshape(at::IntArrayRef(axis_mask, numDimensions)).expand(X_shape);
  auto zero_point_vectorized = zero_point_rounded.reshape(at::IntArrayRef(axis_mask, numDimensions)).expand(X_shape);

  auto iter = TensorIteratorConfig()
    .add_output(dX)
    .add_output(dScale_vec)
    .add_output(dZeroPoint_vec)
    .add_input(X)
    .add_input(dY)
    .add_input(scale_vectorized)
    .add_input(zero_point_vectorized)
    .build();

  fake_quant_grad_learnable_channel_stub(
    X.device().type(), iter, quant_min, quant_max, grad_factor);

  auto numElements = X.ndimension() - 1;

  // Create a collection of axes that include all but the channel axis for
  // reduction when summing over the dScale and dZeroPoint tensors.
  int64_t* axis_for_reduction = (int64_t*) calloc(numElements, sizeof(int64_t));
  for (const auto i : c10::irange(axis)) {
    axis_for_reduction[i] = i;
  }
  for (const auto i : c10::irange(axis, numElements)) {
    axis_for_reduction[i] = i + 1;
  }

  auto dScale = dScale_vec.sum(at::IntArrayRef(axis_for_reduction, numElements));
  auto dZeroPoint = dZeroPoint_vec.sum(at::IntArrayRef(axis_for_reduction, numElements));

  free(axis_mask);
  free(axis_for_reduction);
  return std::make_tuple(dX, dScale, dZeroPoint);
}
} // namespace native
} // namespace at
