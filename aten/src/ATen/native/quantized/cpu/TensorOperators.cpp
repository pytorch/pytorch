#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/core/QScheme.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/eq.h>
#include <ATen/ops/eq_native.h>
#include <ATen/ops/ge.h>
#include <ATen/ops/ge_native.h>
#include <ATen/ops/gt.h>
#include <ATen/ops/gt_native.h>
#include <ATen/ops/le.h>
#include <ATen/ops/le_native.h>
#include <ATen/ops/lt.h>
#include <ATen/ops/lt_native.h>
#include <ATen/ops/ne.h>
#include <ATen/ops/ne_native.h>
#include <ATen/ops/resize_native.h>
#endif

namespace at {
namespace native {

/*
All comparator operators will be named "<aten op name>_quantized_cpu".
'_out' will be appended for the 'out' variant of the op.

TODO: This is an inefficient implementation that uses `.dequantize`.
      Need a more efficient implementation.
*/

#define DEFINE_COMPARATOR(at_op) \
Tensor& at_op##_out_quantized_cpu(const Tensor& self, \
                                const Scalar& other, Tensor& out) { \
  TORCH_CHECK(out.dtype() == at::ScalarType::Bool, \
              "The 'out' tensor must have dtype 'torch.bool'"); \
  auto self_dq = self.dequantize(); \
  return at:: at_op##_out(out, self_dq, other); \
} \
Tensor at_op##_quantized_cpu(const Tensor& self, const Scalar& other) { \
  auto self_dq = self.dequantize(); \
  return at:: at_op(self_dq, other); \
} \
Tensor& at_op##_out_quantized_cpu(const Tensor& self, \
                                const Tensor& other, Tensor& out) { \
  /* We infer size to make sure the tensors are compatible. */\
  infer_size_dimvector(self.sizes(), other.sizes()); \
  TORCH_CHECK(out.dtype() == at::ScalarType::Bool, \
              "The 'out' tensor must have dtype 'torch.bool'"); \
  auto self_dq = self.dequantize(); \
  auto other_dq = other.dequantize(); \
  return at:: at_op##_out(out, self_dq, other_dq); \
} \
Tensor at_op##_quantized_cpu(const Tensor& self, const Tensor& other) { \
  /* We infer size to make sure the tensors are compatible. */\
  infer_size_dimvector(self.sizes(), other.sizes()); \
  auto self_dq = self.dequantize(); \
  auto other_dq = other.dequantize(); \
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

const Tensor& quantized_resize_cpu_(
    const Tensor& self,
    IntArrayRef size,
    c10::optional<MemoryFormat> optional_memory_format) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because if storage is resized, new elements are uninitialized
  globalContext().alertNotDeterministic("quantized_resize_cpu_");
  TORCH_CHECK(
      !optional_memory_format.has_value(),
      "Unsupported memory format for quantized tensor resize ",
      optional_memory_format.value());
  auto qscheme = self.quantizer()->qscheme();
  TORCH_CHECK(
      qscheme == QScheme::PER_TENSOR_AFFINE ||
          qscheme == QScheme::PER_TENSOR_SYMMETRIC,
      "Can only resize quantized tensors with per-tensor schemes!");
  auto* self_ = self.unsafeGetTensorImpl();
  // NOLINTNEXTLINE(bugprone-argument-comment)
  resize_impl_cpu_(self_, size, /*strides=*/c10::nullopt);
  return self;
}

}}  // at::native
