#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/EmptyTensor.h>
#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_native.h>
#include <ATen/ops/empty_strided_native.h>
#endif

namespace at::native {

Tensor empty_meta_symint(
  SymIntArrayRef size,
  c10::optional<ScalarType> dtype_opt,
  c10::optional<Layout> layout_opt,
  c10::optional<Device> device_opt,
  c10::optional<bool> pin_memory_opt,
  c10::optional<c10::MemoryFormat> memory_format_opt
) {

  auto opt_size = asIntArrayRefSlowOpt(size);
  if (opt_size.has_value()) {
    return at::detail::empty_meta(*opt_size, dtype_opt, layout_opt, device_opt, pin_memory_opt, memory_format_opt);
  }
  return at::detail::empty_symint_meta(
      size, dtype_opt, layout_opt, device_opt, pin_memory_opt, memory_format_opt);
}

// Kept only for BC with XLA
static Tensor empty_strided_meta(
  IntArrayRef size,
  IntArrayRef stride,
  c10::optional<ScalarType> dtype_opt,
  c10::optional<Layout> layout_opt,
  c10::optional<Device> device_opt,
  c10::optional<bool> pin_memory_opt
) {
  return empty_strided_meta_symint(c10::fromIntArrayRefSlow(size), c10::fromIntArrayRefSlow(stride), dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

Tensor empty_strided_meta_symint(
  SymIntArrayRef size,
  SymIntArrayRef stride,
  c10::optional<ScalarType> dtype_opt,
  c10::optional<Layout> layout_opt,
  c10::optional<Device> device_opt,
  c10::optional<bool> pin_memory_opt
) {
  return at::detail::empty_strided_symint_meta(
      size, stride, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

} // namespace at::native
