#include <ATen/ATen.h>
#include <ATen/core/Type.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/quantized/Quantizer.h>

#include <algorithm>

namespace at { namespace native {
namespace {

class QReluInt8 final : public c10::OperatorKernel {
 public:
  Tensor operator()(Tensor qx) {
    Tensor qy = at::_empty_affine_quantized(qx.sizes(),
                                            at::device(kCPU).dtype(kQInt8),
                                            qx.q_scale().toDouble(),
                                            qx.q_zero_point().toLong());
    auto iter = TensorIterator::unary_op(qy, qx);
    const auto zero_point = qx.q_zero_point().toByte();
    unary_kernel(*iter, [&](c10::qint8 value) -> c10::qint8 {
      return c10::qint8(std::max(value.val_, zero_point));
    });
    return qy;
  }
};

static auto registry = c10::RegisterOperators().op(
    "quantized::relu(Tensor qx) -> Tensor",
    c10::kernel<QReluInt8>(),
    c10::dispatchKey(QuantizedCPUTensorId()));

} // namespace
}} // namespace at::native
