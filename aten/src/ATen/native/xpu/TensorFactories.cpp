#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/xpu/EmptyTensor.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_native.h>
#include <ATen/ops/empty_strided_native.h>
#endif

namespace at::native {

Tensor empty_xpu(IntArrayRef size, c10::optional<ScalarType> dtype_opt, c10::optional<Layout> layout_opt, c10::optional<Device> device_opt, c10::optional<bool> pin_memory_opt, c10::optional<c10::MemoryFormat> memory_format_opt) {
  Tensor result = at::detail::empty_xpu(size, dtype_opt, layout_opt, device_opt, pin_memory_opt, memory_format_opt);
  // See Note [Enabling Deterministic Operations]
  TORCH_CHECK(!(C10_UNLIKELY(at::globalContext().deterministicAlgorithms() && at::globalContext().deterministicFillUninitializedMemory())), "XPU backend doesn't support deterministic implementation for empty ...")
  return result;
}

Tensor empty_strided_xpu(IntArrayRef size, IntArrayRef stride, c10::optional<ScalarType> dtype_opt, c10::optional<Layout> layout_opt, c10::optional<Device> device_opt, c10::optional<bool> pin_memory_opt) {
  Tensor result = at::detail::empty_strided_xpu(size, stride, dtype_opt, layout_opt, device_opt, pin_memory_opt);
  // See Note [Enabling Deterministic Operations]
  TORCH_CHECK(!(C10_UNLIKELY(at::globalContext().deterministicAlgorithms() && at::globalContext().deterministicFillUninitializedMemory())), "XPU backend doesn't support deterministic implementation for empty_strided ...")
  return result;
}

TORCH_LIBRARY_IMPL(aten, XPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::empty.memory_format"), TORCH_FN(at::native::empty_xpu));
  m.impl(TORCH_SELECTIVE_NAME("aten::empty_strided"), TORCH_FN(at::native::empty_strided_xpu));
}

} // namespace at::native
