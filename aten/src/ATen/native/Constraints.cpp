#include <limits>
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
#include <ATen/ops/sym_constrain_range_for_size_native.h>
#include <ATen/ops/_functional_sym_constrain_range_for_size_native.h>
#endif

namespace at {
namespace native {

void sym_constrain_range(
    const Scalar& size,
    c10::optional<int64_t> min,
    c10::optional<int64_t> max) {

    int64_t min_val = min.has_value() ? min.value() : std::numeric_limits<int64_t>::min();
    int64_t max_val = max.has_value() ? max.value() : std::numeric_limits<int64_t>::max();
    int64_t size_as_int = size.toInt();

    TORCH_CHECK(
      max_val >= min_val,
      "Max must be greater than or equal to min. Got min=",
      min_val,
      " max=",
      max_val
    );

    TORCH_CHECK(
      min_val <= size_as_int && size_as_int <= max_val,
      "Invalid value range for ",
      size_as_int,
      " between [",
      min_val,
      ", ",
      max_val,
      "]."
    );
}

Tensor _functional_sym_constrain_range(
    const Scalar& size,
    c10::optional<int64_t> min,
    c10::optional<int64_t> max,
    const Tensor& dep_token) {
  sym_constrain_range(size, min, max);
  return dep_token.clone();
}

void sym_constrain_range_for_size(const Scalar& size, c10::optional<int64_t> min, c10::optional<int64_t> max) {
  int64_t min_val = min.has_value() ? min.value() : 0;
  if (max.has_value() && max.value() <= 2) {
    TORCH_CHECK(false, "Max value to constrain_range_for_size must be greater than 2. got: ", max.value());
  }
  sym_constrain_range(size, min_val, max);
}

Tensor _functional_sym_constrain_range_for_size(
  const Scalar& size,
  c10::optional<int64_t> min,
  c10::optional<int64_t> max,
  const Tensor& dep_token) {
  sym_constrain_range_for_size(size, min, max);
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
