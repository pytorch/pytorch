#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/quantized/cpu/quantized_ops.h>
#include <ATen/quantized/Quantizer.h>

#include <algorithm>

namespace at {
namespace native {

DEFINE_DISPATCH(qclamp_stub);

namespace {
Tensor quantized_clamp(const Tensor& qx, Scalar min, Scalar max) {
  Tensor qy;
  qclamp_stub(qx.device().type(), qx, min, max, qy);
  return qy;
}

class QClamp final : public c10::OperatorKernel {
 public:
  Tensor operator()(Tensor qx, Scalar min, Scalar max) {
    return quantized_clamp(qx, min, max);
  }
};

static auto registry = c10::RegisterOperators().op(
    "quantized::clamp(Tensor qx, Scalar min, Scalar max) -> Tensor qy",
    c10::RegisterOperators::options().kernel<QClamp>(
        TensorTypeId::QuantizedCPUTensorId));
} // namespace

} // namespace native
} // namespace at
