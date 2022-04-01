#include <torch/library.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/PythonModeTLS.h>

namespace {

// TLS saving the state of the include/exclude sets on entry to the dispatcher
// This is set in the pythonTLSSnapshot fallback and used by the Python fallback.
thread_local c10::optional<c10::impl::LocalDispatchKeySet> tls_on_entry;

void pythonFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  TORCH_INTERNAL_ASSERT(tls_on_entry.has_value());
  c10::impl::ForceDispatchKeyGuard guard(tls_on_entry.value());

  // If Python Mode is active, use its PyInterpreter for dispatch
  const auto& maybe_python_mode_state = at::impl::PythonModeTLS::get_state();
  if (maybe_python_mode_state) {
    maybe_python_mode_state->pyinterpreter()->dispatch(op, stack, maybe_python_mode_state);
    return;
  }

  // Otherwise, find a PyInterpreter on a Tensor
  const auto& schema = op.schema();
  const auto num_arguments = schema.arguments().size();
  // It is safe to dispatch on the very first Tensor with a pyobj_interpreter
  // without checking the interpreters of any of the arguments, because when
  // we actually run dispatch(), we will take out PyObjects in the context
  // of that interpreter, and this will ensure that everyone is on the same
  // interpreter.
  for (const auto& ivalue : torch::jit::last(*stack, num_arguments)) {
    if (ivalue.isTensor()) {
      auto* interpreter = ivalue.unsafeToTensorImpl()->pyobj_interpreter();
      if (interpreter) {
        interpreter->dispatch(op, stack, nullptr);
        return;
      }
    } else if (ivalue.isTensorList()) {
      // NB: use toListRef as it doesn't induce refcount bumps (toTensorListRef
      // is not a thing)
      for (const auto& nv : ivalue.toListRef()) {
        auto* interpreter = nv.unsafeToTensorImpl()->pyobj_interpreter();
        if (interpreter) {
          interpreter->dispatch(op, stack, nullptr);
          return;
        }
      }
    }
  }
  TORCH_INTERNAL_ASSERT(0, "Hit Python dispatch key but no arguments had PyInterpreter (no tensor args?)");
}

void pythonTLSSnapshotFallback(const c10::OperatorHandle& op, c10::DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
  // It is ok for the tls to be already set here.
  // A CompositeImplicitAutograd function may have been called just before this and so the tls here were never cleared
  // This is also why we don't need an RAII to ensure the tls is reset when exceptions happen

  tls_on_entry = c10::impl::tls_local_dispatch_key_set();

  op.redispatchBoxed(dispatch_keys & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::PythonTLSSnapshot), stack);

  tls_on_entry = c10::nullopt;
}


} // anonymous namespace

TORCH_LIBRARY_IMPL(_, Python, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&pythonFallback>());
}

TORCH_LIBRARY_IMPL(_, PythonTLSSnapshot, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&pythonTLSSnapshotFallback>());
}
