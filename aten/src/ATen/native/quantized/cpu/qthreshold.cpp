#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/native/quantized/cpu/quantized_ops.h>

#include <algorithm>

namespace at {
namespace native {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(qthreshold_stub);

// the underlying implementation for quantized threshold kernel
Tensor quantized_threshold_impl(
    const Tensor& qx,
    const Scalar& threshold,
    const Scalar& value) {
  Tensor qy = at::_empty_affine_quantized(
    qx.sizes(), qx.options(), qx.q_scale(), qx.q_zero_point());
  qthreshold_stub(qx.device().type(), qx, threshold, value, qy);
  return qy;
}

// at::native functions for the native_functions.yaml
Tensor threshold_quantized_cpu(
    const Tensor& qx,
    const Scalar& threshold,
    const Scalar& value) {
  Tensor qy;
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "threshold", [&]() {
    qy = quantized_threshold_impl(qx, threshold, value);
  });
  return qy;
}

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::threshold"), TORCH_FN(threshold_quantized_cpu));
}

} // namespace native
} // namespace at
