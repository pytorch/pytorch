#include <ATen/PythonMode.h>

namespace at { namespace impl {

thread_local optional<TorchDispatchOverrideImpl> torchDispatchOverride;

const optional<TorchDispatchOverrideImpl>& PythonMode::get_torch_dispatch() {
  return torchDispatchOverride;
}

void PythonMode::set_torch_dispatch(const TorchDispatchOverrideImpl& dispatch_override) {
  TORCH_CHECK(
      !torchDispatchOverride.has_value(),
      "enable_factory_dispatch has already been set. Please reset it before setting it again.")
  torchDispatchOverride = dispatch_override;
  c10::impl::tls_set_dispatch_key_included(DispatchKey::PythonMode, true);
}

void PythonMode::set_torch_dispatch_with_tensor(const Tensor& tensor) {
  PythonMode::set_torch_dispatch(tensor);
}

void PythonMode::reset_torch_dispatch() {
  torchDispatchOverride = nullopt;
  c10::impl::tls_set_dispatch_key_included(DispatchKey::PythonMode, false);
}

TORCH_LIBRARY_IMPL(_, PythonMode, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

void dispatchFactoryToPython(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  TORCH_INTERNAL_ASSERT(torchDispatchOverride.has_value());
  auto* interpreter = torchDispatchOverride->tensor().unsafeGetTensorImpl()->pyobj_interpreter();
  TORCH_INTERNAL_ASSERT(interpreter);

  // Invariant: the following only works if stack has no dispatch-able tensors!
  // This is OK, because `op` should be a factory function that accepts no Tensors.
  interpreter->dispatch(op, stack, &torchDispatchOverride.value());
}

// TORCH_LIBRARY_IMPL(aten, PythonMode, m) {
//   m.impl("empty.memory_format", torch::CppFunction::makeFromBoxedFunction<&dispatchFactoryToPython>());
// }

}} // namespace at::impl
