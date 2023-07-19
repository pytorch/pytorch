#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>
#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_functional_sym_constrain_range_native.h>
#include <ATen/ops/_make_dep_token_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/sym_constrain_range_native.h>
#endif

namespace at {
namespace native {

void sym_constrain_range_cpu(
    const Scalar& size,
    c10::optional<int64_t> min,
    c10::optional<int64_t> max) {}

Tensor _functional_sym_constrain_range_cpu(
    const Scalar& size,
    c10::optional<int64_t> min,
    c10::optional<int64_t> max,
    const Tensor& dep_token) {
  return dep_token.clone();
}

Tensor _make_dep_token_cpu(
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt) {
  return at::empty(
      {}, dtype_opt, layout_opt, device_opt, pin_memory_opt, memory_format_opt);
}

} // namespace native
} // namespace at
