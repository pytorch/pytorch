#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/native/quantized/cpu/quantized_ops.h>

#include <algorithm>

namespace at {
namespace native {

DEFINE_DISPATCH(qhardswish_stub);

Tensor quantized_hardswish(const Tensor& qx) {
  Tensor qy = at::_empty_affine_quantized(qx.sizes(), qx.options(),
      qx.q_scale(), qx.q_zero_point());
  qhardswish_stub(qx.device().type(), qx, qy);
  return qy;
}

Tensor& quantized_hardswish_(Tensor& qx) {
  qhardswish_stub(qx.device().type(), qx, qx);
  return qx;
}

}}  // namespace at::native
