#pragma once

#include <ATen/Tensor.h>

// special op invocation pattern from: ATen/native/c10_utils.h
template <class... Inputs>
inline std::vector<c10::IValue> makeStack(Inputs&&... inputs) {
  return {std::forward<Inputs>(inputs)...};
}

// Note: this is slow because it boxes every argument
// and likely has to re-unbox them when calling the kernel.
// Prefer calling c10::OperatorHandle::callUnboxed<Args...>(args...).
template <class... Args>
inline std::vector<c10::IValue> call_unboxed_super_slow_temp_shim(
    const c10::OperatorHandle& op,
    Args... args) {
  auto stack = makeStack(std::forward<Args>(args)...);
  op.callBoxed(&stack);
  return stack;
}

// Note: this is slow because it boxes every argument
// and likely has to re-unbox them when calling the kernel.
// Prefer calling c10::OperatorHandle::callUnboxed<Args...>(args...).
template <class... Args>
inline std::vector<c10::IValue> call_unboxed_super_slow_temp_shim(
    const char* func_name,
    const char* overload_name,
    Args... args) {
  const c10::optional<c10::OperatorHandle> op_handle =
      c10::Dispatcher::singleton().findSchema({func_name, overload_name});
  assert(op_handle.has_value());
  return callOp(op_handle.value(), args...);
}
