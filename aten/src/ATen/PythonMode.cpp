#include <ATen/PythonMode.h>
#include <torch/library.h>
#include <ATen/core/boxing/KernelFunction.h>
#include <ATen/core/dispatch/Dispatcher.h>

namespace at { namespace impl {

// NB: NOT thread safe
optional<Tensor> torchDispatchFuncTensor;

void PythonMode::set_torch_dispatch(const Tensor& tensor) {
  TORCH_CHECK(
      !torchDispatchFuncTensor.has_value(),
      "PythonMode has already been set. Please reset it before setting it again.")
  torchDispatchFuncTensor = tensor;
  c10::impl::tls_set_dispatch_key_included(DispatchKey::PythonMode, true);
}

void PythonMode::reset_torch_dispatch() {
  torchDispatchFuncTensor = nullopt;
  c10::impl::tls_set_dispatch_key_included(DispatchKey::PythonMode, false);
}

TORCH_LIBRARY_IMPL(_, PythonMode, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

static void dispatchFactoryToPython(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  TORCH_INTERNAL_ASSERT(torchDispatchFuncTensor.has_value());
  auto* interpreter = torchDispatchFuncTensor->unsafeGetTensorImpl()->pyobj_interpreter();
  TORCH_INTERNAL_ASSERT(interpreter);
  interpreter->dispatch(op, stack, &torchDispatchFuncTensor.value());
}

TORCH_LIBRARY_IMPL(aten, PythonMode, m) {
  m.impl("empty.memory_format", torch::CppFunction::makeFromBoxedFunction<&dispatchFactoryToPython>());
}

}} // namespace at::impl
