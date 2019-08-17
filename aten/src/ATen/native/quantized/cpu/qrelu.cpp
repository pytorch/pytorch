#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/quantized/Quantizer.h>

#include <algorithm>

namespace at {
namespace native {
Tensor quantized_relu(const Tensor& qx) {
  Tensor qy;
  const auto zero_point = qx.q_zero_point();
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qrelu", [&]() {
    qy = at::_empty_affine_quantized(
        qx.sizes(),
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

namespace {
Tensor quantized_relu6(const Tensor& qx) {
  Tensor qy;
  const auto zero_point = qx.q_zero_point();
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qrelu", [&]() {
    qy = at::_empty_affine_quantized(
        qx.sizes(),
        at::device(kCPU).dtype(SCALAR_TYPE),
        qx.q_scale(),
        qx.q_zero_point());
    auto iter = TensorIterator::unary_op(qy, qx);
    scalar_t six = at::quantize_val<scalar_t>(qx.q_scale(), qx.q_zero_point(),
                                              6.0);
    cpu_kernel(iter, [&](scalar_t value) -> scalar_t {
      underlying_t relu_val = std::max<underlying_t>(value.val_, zero_point);
      return scalar_t(std::min<underlying_t>(relu_val, six.val_));
    });
  });
  return qy;
}

class QRelu final : public c10::OperatorKernel {
 public:
  Tensor operator()(Tensor qx) {
    return at::relu(qx);
  }
};

class QRelu6 final : public c10::OperatorKernel {
 public:
  Tensor operator()(Tensor qx) {
    return quantized_relu6(qx);
  }
};

static auto registry = c10::RegisterOperators()
.op("quantized::relu(Tensor qx) -> Tensor",
    c10::RegisterOperators::options().kernel<QRelu>(QuantizedCPUTensorId()))
.op("quantized::relu6(Tensor qx) -> Tensor",
    c10::RegisterOperators::options().kernel<QRelu6>(QuantizedCPUTensorId()));
} // namespace

} // namespace native
} // namespace at
