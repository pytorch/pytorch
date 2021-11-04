// We register ops with a higher priority dispatch key (BackendSelect) than the usual backend-specific keys (e.g. CPU)
// which makes calls to the factory functions dispatch to here.
// We then 'manually' compute a lower-priority to re-dispatch to (e.g. CPU) to get to the eventually correct backend.
// ${generated_comment}

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Operators.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <c10/core/TensorOptions.h>

namespace at {

namespace {

${backend_select_method_definitions}

bool is_pinned(const Tensor& self, c10::optional<at::Device> device) {
  // Only CPU tensors can be pinned
  if (!self.is_cpu()) {
    return false;
  }
  static auto op = c10::Dispatcher::singleton()
    .findSchemaOrThrow("aten::is_pinned", "")
    .typed<bool (const Tensor&, c10::optional<at::Device>)>();
  // TODO: fetch scalar type from Tensor? But it doesn't really matter...
  DispatchKeySet _dk = c10::DispatchKeySet(c10::computeDispatchKey(c10::nullopt, self.layout(), device.value_or(at::kCUDA)));
  return op.redispatch(_dk, self, device);
}

at::Tensor _pin_memory(const Tensor& self, c10::optional<at::Device> device) {
  TORCH_CHECK(self.device().is_cpu(), "cannot pin '", self.toString(), "' only dense CPU tensors can be pinned");
  static auto op = c10::Dispatcher::singleton()
    .findSchemaOrThrow("aten::_pin_memory", "")
    .typed<Tensor (const Tensor&, c10::optional<at::Device>)>();
  DispatchKeySet _dk = c10::DispatchKeySet(c10::computeDispatchKey(c10::nullopt, self.layout(), device.value_or(at::kCUDA)));
  return op.redispatch(_dk, self, device);
}

TORCH_LIBRARY_IMPL(aten, BackendSelect, m) {
  ${backend_select_function_registrations};
  m.impl(TORCH_SELECTIVE_NAME("aten::is_pinned"), TORCH_FN(is_pinned));
  m.impl(TORCH_SELECTIVE_NAME("aten::_pin_memory"), TORCH_FN(_pin_memory));
}

} // namespace
} // at
