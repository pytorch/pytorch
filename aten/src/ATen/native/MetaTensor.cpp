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
  std::optional<ScalarType> dtype_opt,
  std::optional<Layout> layout_opt,
  std::optional<Device> device_opt,
  std::optional<bool> pin_memory_opt,
  std::optional<c10::MemoryFormat> memory_format_opt
) {

  auto opt_size = asIntArrayRefSlowOpt(size);
  if (opt_size.has_value()) {
    return at::detail::empty_meta(*opt_size, dtype_opt, layout_opt, device_opt, pin_memory_opt, memory_format_opt);
  }
  return at::detail::empty_symint_meta(
      size, dtype_opt, layout_opt, device_opt, pin_memory_opt, memory_format_opt);
}

Tensor empty_strided_meta_symint(
  SymIntArrayRef size,
  SymIntArrayRef stride,
  std::optional<ScalarType> dtype_opt,
  std::optional<Layout> layout_opt,
  std::optional<Device> device_opt,
  std::optional<bool> pin_memory_opt
) {
  return at::detail::empty_strided_symint_meta(
      size, stride, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

} // namespace at::native
