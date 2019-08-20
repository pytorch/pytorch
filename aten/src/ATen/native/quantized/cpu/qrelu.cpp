#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/NativeFunctions.h>

#include <algorithm>

namespace at { namespace native {
Tensor quantized_relu(const Tensor& qx) {
  Tensor qy;
  const auto zero_point = qx.q_zero_point();
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qrelu", [&]() {
    qy = at::_empty_affine_quantized(qx.sizes(),
                                     at::device(kCPU).dtype(SCALAR_TYPE),
                                     qx.q_scale(),
                                     qx.q_zero_point());
    auto iter = TensorIterator::unary_op(qy, qx);
    cpu_kernel(iter, [&](scalar_t value) -> scalar_t {
      return scalar_t(std::max<underlying_t>(value.val_, zero_point));
    });
  });
  return qy;
}

Tensor& quantized_relu_(Tensor& qx) {
  const auto zero_point = qx.q_zero_point();
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qrelu", [&]() {
    auto iter = TensorIterator::unary_op(qx, qx);
    cpu_kernel(iter, [&](scalar_t value) -> scalar_t {
      return scalar_t(std::max<underlying_t>(value.val_, zero_point));
    });
  });
  return qx;
}

}}  // namespace at::native
