#include <ATen/core/PythonFallbackKernel.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

namespace {
void dtensorFallback(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet dispatch_keys,
    torch::jit::Stack* stack) {
  at::impl::pythonFallback(op, dispatch_keys, stack);
}
} // namespace

TORCH_LIBRARY_IMPL(_, DTensor, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&dtensorFallback>());
}
