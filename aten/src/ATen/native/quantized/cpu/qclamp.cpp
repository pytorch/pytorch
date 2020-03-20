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
Tensor quantized_clamp_impl(
    const Tensor& qx,
    optional<Scalar> min,
    optional<Scalar> max) {
  Tensor qy;
  if (min && max) {
    qclamp_stub(qx.device().type(), qx, *min, *max, qy);
  } else {
    TORCH_CHECK(
        false, "Both min and max should be specified for quantized clamp!");
  }
  return qy;
}
} // namespace

// at::native functions for the native_functions.yaml
Tensor quantized_clamp(
    const Tensor& qx,
    optional<Scalar> min,
    optional<Scalar> max) {
  Tensor qy;
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "clamp", [&]() {
    qy = quantized_clamp_impl(qx, min, max);
  });
  return qy;
}

// hardtanh is clamp with default min==-1.0f and default max==1.0f
Tensor quantized_hardtanh(
    const Tensor& qx,
    Scalar min,
    Scalar max) {
  Tensor qy;
  qy = quantized_clamp_impl(qx, min, max);
  return qy;
}

Tensor& quantized_hardtanh_out(
    Tensor& result,
    const Tensor& qx,
    Scalar min,
    Scalar max) {
  result = quantized_clamp_impl(qx, min, max);
  return result;
}

Tensor& quantized_hardtanh_(
    Tensor& self,
    Scalar min,
    Scalar max) {
  Tensor qy;
  qy = quantized_clamp_impl(self, min, max);
  // This can be optimized in a future PR if it becomes a bottleneck.
  self.copy_(qy);
  return self;
}

// Keep the registry in the anonymous namespace.
namespace {
class QClamp final : public c10::OperatorKernel {
 public:
  Tensor operator()(Tensor qx, optional<Scalar> min, optional<Scalar> max) {
    return quantized_clamp(qx, min, max);
  }
};

static auto registry = c10::RegisterOperators().op(
    "quantized::clamp(Tensor qx, Scalar? min, Scalar? max) -> Tensor qy",
    c10::RegisterOperators::options()
        .aliasAnalysis(at::AliasAnalysisKind::FROM_SCHEMA)
        .kernel<QClamp>(DispatchKey::QuantizedCPUTensorId));
} // namespace

} // namespace native
} // namespace at
