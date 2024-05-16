// We register ops with a higher priority dispatch key (BackendSelect) than the usual backend-specific keys (e.g. CPU)
// which makes calls to the factory functions dispatch to here.
// We then 'manually' compute a lower-priority to re-dispatch to (e.g. CPU) to get to the eventually correct backend.
// ${generated_comment}

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/core/dispatch/DispatchKeyExtractor.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Operators.h>
#else
#include <ATen/ops/is_pinned_ops.h>
#include <ATen/ops/_pin_memory_ops.h>

${ops_headers}
#endif

namespace at {

namespace {

${backend_select_method_definitions}

bool is_pinned(const Tensor& self, std::optional<at::Device> device) {
  // Only CPU tensors can be pinned
  if (!self.is_cpu()) {
    return false;
  }
  // TODO: fetch scalar type from Tensor? But it doesn't really matter...
  DispatchKeySet _dk = c10::DispatchKeySet(c10::computeDispatchKey(c10::nullopt, self.layout(), device.value_or(at::kCUDA)));
  return at::_ops::is_pinned::redispatch(_dk, self, device);
}

at::Tensor _pin_memory(const Tensor& self, std::optional<at::Device> device) {
  TORCH_CHECK(self.device().is_cpu(), "cannot pin '", self.toString(), "' only dense CPU tensors can be pinned");
  DispatchKeySet _dk = c10::DispatchKeySet(c10::computeDispatchKey(c10::nullopt, self.layout(), device.value_or(at::kCUDA)));
  if (self.is_nested()) {
    constexpr auto nested_key_set = c10::DispatchKeySet(
        {c10::DispatchKey::NestedTensor, c10::DispatchKey::AutogradNestedTensor});
    _dk = _dk.add(self.key_set() & nested_key_set);
  }
  return at::_ops::_pin_memory::redispatch(_dk, self, device);
}

TORCH_LIBRARY_IMPL(aten, BackendSelect, m) {
  ${backend_select_function_registrations};
  m.impl(TORCH_SELECTIVE_NAME("aten::is_pinned"), TORCH_FN(is_pinned));
  m.impl(TORCH_SELECTIVE_NAME("aten::_pin_memory"), TORCH_FN(_pin_memory));
}

} // namespace
} // at
