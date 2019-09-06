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

DEFINE_DISPATCH(qrelu_stub);
DEFINE_DISPATCH(qrelu6_stub);

Tensor quantized_relu(const Tensor& qx) {
  Tensor qy;
  qrelu_stub(qx.device().type(), qx, qy);
  return qy;
}
Tensor& quantized_relu_(Tensor& qx) {
  const auto zero_point = qx.q_zero_point();
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qrelu", [&]() {
    using Vec = Vec256<scalar_t>;
    auto iter = TensorIterator::unary_op(qx, qx);
    auto zero_point_vec = Vec(scalar_t(zero_point));
    cpu_kernel_vec(
        iter,
        [&](scalar_t value) -> scalar_t {
          return scalar_t(std::max<underlying_t>(value.val_, zero_point));
        },
        [&](Vec value) -> Vec { return value.relu(zero_point_vec); });
  });
  return qx;
}

namespace {
Tensor quantized_relu6(const Tensor& qx) {
  Tensor qy;
  qrelu6_stub(qx.device().type(), qx, qy);
  return qy;
}

class QRelu6 final : public c10::OperatorKernel {
 public:
  Tensor operator()(Tensor qx) {
    return quantized_relu6(qx);
  }
};

static auto registry = c10::RegisterOperators()
.op("quantized::relu6(Tensor qx) -> Tensor",
    c10::RegisterOperators::options().kernel<QRelu6>(TensorTypeId::QuantizedCPUTensorId));
} // namespace

}}  // namespace at::native
