#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <torch/library.h>
#include <ATen/native/quantized/cpu/QuantizedOps.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/threshold_native.h>
#endif

#include <algorithm>

namespace at {
namespace native {

DEFINE_DISPATCH(qthreshold_stub);

// the underlying implementation for quantized threshold kernel
static Tensor quantized_threshold_impl(
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
