#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>

namespace at {
namespace native {

/*
All comparator operators will be named "quantized_<aten op name>" and
"quantized_<aten op name>_out".
*/

#define DEFINE_COMPARATOR(at_op) \
Tensor& quantized_##at_op##_out(Tensor& out, const Tensor& self, \
                                Scalar other) { \
  TORCH_CHECK(out.dtype() == at::ScalarType::Bool, \
              "The 'out' tensor must have dtype 'torch.bool'"); \
  const auto& self_dq = self.dequantize(); \
  return at:: at_op##_out(out, self_dq, other); \
} \
Tensor quantized_##at_op(const Tensor& self, Scalar other) { \
  const auto& self_dq = self.dequantize(); \
  return at:: at_op(self_dq, other); \
} \
Tensor& quantized_##at_op##_out(Tensor& out, const Tensor& self, \
                                const Tensor& other) { \
  infer_size(self.sizes(), other.sizes()); \
  TORCH_CHECK(out.dtype() == at::ScalarType::Bool, \
              "The 'out' tensor must have dtype 'torch.bool'"); \
  const auto& self_dq = self.dequantize(); \
  const auto& other_dq = other.dequantize(); \
  return at:: at_op##_out(out, self_dq, other_dq); \
} \
Tensor quantized_##at_op(const Tensor& self, const Tensor& other) { \
  infer_size(self.sizes(), other.sizes()); \
  const auto& self_dq = self.dequantize(); \
  const auto& other_dq = other.dequantize(); \
  return at:: at_op(self_dq, other_dq); \
}

#define AT_FORALL_OPERATORS(_) \
_(ne)                          \
_(eq)                          \
_(ge)                          \
_(le)                          \
_(gt)                          \
_(lt)                          \

AT_FORALL_OPERATORS(DEFINE_COMPARATOR)

#undef AT_FORALL_OPERATORS
#undef DEFINE_COMPARATOR

}}  // at::native
