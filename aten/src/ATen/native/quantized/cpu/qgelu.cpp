#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/native/quantized/cpu/quantized_ops.h>

#include <algorithm>

namespace at {
namespace native {

DEFINE_DISPATCH(qgelu_stub);

Tensor quantized_gelu(const Tensor& qx) {
  Tensor qy;
  qy = at::_empty_affine_quantized(
      qx.sizes(),
      qx.options(),
      qx.q_scale(),
      qx.q_zero_point());
  qgelu_stub(qx.device().type(), qx, qy);
  return qy;
}
}}  // namespace at::native
