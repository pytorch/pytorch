#pragma once

#include <stack>

#include <c10/core/InferenceMode.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/util/Exception.h>
#include <c10/util/ThreadLocalDebugInfo.h>

#include <ATen/record_function.h>
#include <ATen/FuncTorchTLS.h>
#include <ATen/core/PythonModeTLS.h>

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

  std::shared_ptr<TorchDispatchTypeObject> python_mode_state_;

  // TLS for saved tensors default hooks
  std::stack<std::pair<PyObject*, PyObject*>> saved_tensors_default_hooks_;

  // Whether pre-sampling RecordFunction optimization was enabled
  bool bumped_record_all_functions_ = false;

  friend class ThreadLocalStateGuard;
};

// Guard to set and reset the thread local state
class TORCH_API ThreadLocalStateGuard {
 public:
  explicit ThreadLocalStateGuard(const ThreadLocalState& state)
      : prev_state_(ThreadLocalState()),
        bumped_record_all_functions_(state.bumped_record_all_functions_) {
    // Special handling of RecordFunction pre-sampling optimization:
    // pre-samping is enabled (bumped) when there're non-sampled
    // (or high-frequency) global or TLS callbacks.
    //
    // ThreadLocalStateGuard simply resets RecordFunction's TLS and
    // hence its thread local callbacks.
    //
    // Checking if the pre-sampling was enabled and preserving it in the
    // async task by calling bumpRecordAllFunctions() and the corresponding
    // releaseRecordAllFunctions()
    if (bumped_record_all_functions_) {
      at::bumpRecordAllFunctions();
    }
    // set the given state across the thread boundary
    ThreadLocalState::setThreadLocalState(state);
  }

  ~ThreadLocalStateGuard() {
    // restore previously set variables
    ThreadLocalState::setThreadLocalState(prev_state_);
    if (bumped_record_all_functions_) {
      at::releaseRecordAllFunctions();
    }
  }

 private:
  const ThreadLocalState prev_state_;
  // Whether pre-sampling RecordFunction optimization was enabled
  bool bumped_record_all_functions_ = false;
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
