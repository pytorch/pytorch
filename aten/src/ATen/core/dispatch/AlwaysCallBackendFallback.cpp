#include <torch/library.h>
#include <ATen/core/dispatch/Dispatcher.h>

using c10::Dispatcher;
using c10::DispatchKeySet;
using c10::DispatchKey;

namespace {

void backend_fallback(const c10::OperatorHandle& op, DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
    Dispatcher::singleton().redispatchBoxed(op, dispatch_keys.remove(DispatchKey::AlwaysCall), stack);
}

TORCH_LIBRARY_IMPL(_, AlwaysCall, m) {
   // If there is not a kernel explicitly registered
   // for AutogradXLA, fallthrough to the next
   // available kernel
   m.fallback(torch::CppFunction::makeFromBoxedFunction<&backend_fallback>());
}

}
