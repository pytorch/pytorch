#pragma once

#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/util/Exception.h>
#include <c10/util/ThreadLocalDebugInfo.h>

#include <ATen/record_function.h>

namespace at {

// Thread local state contains values that are preserved across
// thread boundaries (e.g. at::launch/JIT fork, autograd, at::parallel_for)
class TORCH_API ThreadLocalState {
 public:
  // Saves the thread local variables' values and
  // returns them as a ThreadLocalState
  // keep_grad_mode - whether grad mode has to be preserved
  //  (e.g. not preserved when passing from forward pass into
  //   the autograd engine, autograd engine takes care of grad mode)
  ThreadLocalState(bool keep_grad_mode = true);

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

#if !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
  bool keep_grad_mode_ = true;
  bool grad_mode_enabled_;
#endif

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
std::function<T(void)> wrapPropagateTLSState(
    std::function<T(void)> callback) {
  return [tls_state = ThreadLocalState(), callback = std::move(callback)]() {
    ThreadLocalStateGuard g(tls_state);
    // Propagate value returned by callback().
    return callback();
  };
}

} // namespace at
