#pragma once

#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/util/Exception.h>

#include <ATen/ThreadLocalDebugInfo.h>

namespace at {

// Thread local state contains values that are preserved across
// thread boundaries (e.g. at::launch/JIT fork, autograd, at::parallel_for)
class ThreadLocalState {
 public:
  // Saves the thread local variables' values and
  // returns them as a ThreadLocalState
  ThreadLocalState();

  // Sets thread local variables in the current thread,
  // according to the thread boundary specified
  static void setThreadLocalState(const ThreadLocalState& state);

 private:
  c10::impl::LocalDispatchKeySet dispatch_key_;

#if !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
  bool grad_mode_enabled_;
#endif

  // ThreadLocalDebugInfo does not change after being created
  // with DebugInfoGuard
  std::shared_ptr<at::ThreadLocalDebugInfo> debug_info_;

  friend class ThreadLocalStateGuard;
};

// Guard to set and reset the thread local state
class ThreadLocalStateGuard {
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

} // namespace torch
