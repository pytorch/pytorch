#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/impl/GPUTrace.h>
#include <torch/library.h>
#include <iostream>

void CUDASanitizerFallback(const c10::OperatorHandle& op, c10::DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    TORCH_INTERNAL_ASSERT(interp);
    (*interp)->trace_kernel_launch(op, stack);
    op.redispatchBoxed(dispatch_keys & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::CUDASanitizer), stack);
}

TORCH_LIBRARY_IMPL(_, CUDASanitizer, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&CUDASanitizerFallback>());
}
