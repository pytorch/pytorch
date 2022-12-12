#pragma once

#include <stack>

#include <c10/core/InferenceMode.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/util/Exception.h>
#include <c10/util/ThreadLocalDebugInfo.h>

#include <ATen/FuncTorchTLS.h>
#include <ATen/PythonTorchFunctionTLS.h>
#include <ATen/SavedTensorHooks.h>
#include <ATen/record_function.h>
#include <c10/core/impl/PythonDispatcherTLS.h>
#include <c10/core/impl/TorchDispatchModeTLS.h>

namespace at {

// Thread local state contains values that are preserved across
// thread boundaries (e.g. at::launch/JIT fork, autograd).
// Note at::parallel_for doesn't preserve TLS across thread boundaries.
class TORCH_API ThreadLocalState {
 public:
  // Saves the thread local variables' values and
  // returns them as a ThreadLocalState
  ThreadLocalState();

  // set_grad_mode - force the value of the grad mode TLS in
  //  the current state object. This is used for example in the
  //  autograd engine.
  void set_grad_mode(bool enabled);

  // set_multithreading_enabled - force the value of the multithreadinmaximum
  // threads TLS in
  //  the current state object. This is used for example in the
  //  autograd engine.
  void set_multithreading_enabled(bool enabled);

  // Sets thread local variables in the current thread,
  // according to the thread boundary specified
  static void setThreadLocalState(const ThreadLocalState& state);

 private:
  c10::impl::LocalDispatchKeySet dispatch_key_;

  // ThreadLocalDebugInfo does not change after being created
  // with DebugInfoGuard
  std::shared_ptr<c10::ThreadLocalDebugInfo> debug_info_;

  // RecordFunction TLS
  RecordFunctionTLS rf_tls_;

  // TLS for out-of-tree functorch
  // See NOTE [functorch TLS in pytorch/pytorch] for why this needs to be a
  // pointer (spoiler alert: it's due to the indirection)
  // This needs to be a shared_ptr instead of a unique_ptr because
  // ThreadLocalState is copy-able and does indeed get copied. Maybe we can
  // consider adding an explicit copy constructor for ThreadLocalState in the
  // future but I didn't want to add one just for this.
  std::shared_ptr<const functorch::FuncTorchTLSBase> functorch_tls_;

  // TLS for AutogradModes
  AutogradState autograd_tls_;

  // TLS for enable_torch_dispatch_mode
  c10::impl::TorchDispatchModeTLS torch_dispatch_mode_state_;

  // TLS for enable_python_dispatcher
  c10::impl::PyInterpreter* python_dispatcher_state_;

  // TLS for __torch_function__ (mode and disable_torch_function)
  at::impl::PythonTorchFunctionTLS python_torch_function_state_;

  // TLS for saved tensors default hooks
  at::impl::SavedTensorDefaultHooksTLS saved_tensors_default_hooks_state_;

  bool functionalization_reapply_views_state_;

  friend class ThreadLocalStateGuard;
};

// Guard to set and reset the thread local state
class TORCH_API ThreadLocalStateGuard {
 public:
  explicit ThreadLocalStateGuard(const ThreadLocalState& state)
      : prev_state_(ThreadLocalState()) {
    // set the given state across the thread boundary
    ThreadLocalState::setThreadLocalState(state);
  }

  ~ThreadLocalStateGuard() {
    // restore previously set variables
    ThreadLocalState::setThreadLocalState(prev_state_);
  }

 private:
  const ThreadLocalState prev_state_;
};

template <typename T>
auto wrapPropagateTLSState(T callback) {
  return [tls_state = ThreadLocalState(),
          callback = std::move(callback)](auto&&... args) {
    ThreadLocalStateGuard g(tls_state);
    // Propagate value returned by callback().
    return callback(std::forward<decltype(args)>(args)...);
  };
}

} // namespace at
