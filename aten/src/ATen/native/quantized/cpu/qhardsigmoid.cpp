#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/native/quantized/cpu/quantized_ops.h>

#include <algorithm>

namespace at {
namespace native {

DEFINE_DISPATCH(qhardsigmoid_stub);

Tensor quantized_hardsigmoid(const Tensor& qx) {
  Tensor qy;
  qhardsigmoid_stub(qx.device().type(), qx, qy);
  return qy;
}

}}  // namespace at::native
