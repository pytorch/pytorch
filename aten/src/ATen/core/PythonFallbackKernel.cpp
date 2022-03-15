#include <torch/library.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/PythonModeTLS.h>

#include <stack>

namespace {

// This TLS is used to track the state of the dispatcher to be able to restore
// it when calling back into python.
// It has the following invariant:
//  - It must be empty while python code is executed.
//  - It should only be set once even for multiple dispatcher calls that do not come
//    back to python.
// To achieve this, we ensure that the tls is empty by default and emptied again both when
// we call into user torch_dispatch or returning back to python after this call.

thread_local c10::optional<c10::impl::LocalDispatchKeySet> tls_on_entry;

// RAII guard to make working with the above TLS safer.
struct MaybeSetTLSOnEntryGuard {
public:
  MaybeSetTLSOnEntryGuard() {
    if (tls_on_entry.has_value()) {
      value_set_ = false;
    } else {
      value_set_ = true;
      tls_on_entry = c10::impl::tls_local_dispatch_key_set();
    }
  }
  ~MaybeSetTLSOnEntryGuard() {
    if (value_set_) {
      TORCH_INTERNAL_ASSERT(tls_on_entry.has_value());
      tls_on_entry = c10::nullopt;
    }
  }

private:
  bool value_set_;
};

// This guard assumes that tls_on_entry has a value.
struct StashTLSOnEntryGuard {
public:
  StashTLSOnEntryGuard(): saved_(tls_on_entry.value()) {
    tls_on_entry = c10::nullopt;
  }

  ~StashTLSOnEntryGuard() {
    TORCH_INTERNAL_ASSERT(!tls_on_entry.has_value());
    tls_on_entry = saved_;
  }

private:
  c10::impl::LocalDispatchKeySet saved_;
};

void pythonFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  TORCH_INTERNAL_ASSERT(tls_on_entry.has_value());
  c10::impl::ForceDispatchKeyGuard dispatcher_guard(tls_on_entry.value());
  StashTLSOnEntryGuard stash_guard;

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
  // It means that there are multiple calls into the dispatcher not originating from python code.
  // The guard below will properly ignore such calls.
  MaybeSetTLSOnEntryGuard guard;

  op.redispatchBoxed(dispatch_keys & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::PythonTLSSnapshot), stack);
}


} // anonymous namespace

TORCH_LIBRARY_IMPL(_, Python, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&pythonFallback>());
}

TORCH_LIBRARY_IMPL(_, PythonTLSSnapshot, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&pythonTLSSnapshotFallback>());
}
