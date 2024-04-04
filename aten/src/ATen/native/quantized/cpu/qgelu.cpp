#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/quantized/cpu/QuantizedOps.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/gelu_native.h>
#endif

namespace at {
namespace native {

DEFINE_DISPATCH(qgelu_stub);

Tensor gelu_quantized_cpu(const Tensor& qx, c10::string_view approximate) {
  Tensor qy;
  qgelu_stub(qx.device().type(), qx, qy, get_gelutype_enum(approximate));
  return qy;
}

Tensor& gelu_quantized_cpu_(Tensor& self, c10::string_view approximate) {
  Tensor qy = gelu_quantized_cpu(self, approximate);
  // This can be optimized in a future PR if it becomes a bottleneck.
  self.copy_(qy);
  return self;
}

}}  // namespace at::native
