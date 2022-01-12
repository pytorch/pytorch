#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/quantized/cpu/quantized_ops.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/gelu_native.h>
#endif

namespace at {
namespace native {

DEFINE_DISPATCH(qgelu_stub);

Tensor gelu_quantized_cpu(const Tensor& qx) {
  Tensor qy;
  qgelu_stub(qx.device().type(), qx, qy);
  return qy;
}
}}  // namespace at::native
