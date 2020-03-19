#pragma once

namespace torch {

// Thread local state that needs to be propagated across different threads
// during JIT interpreter continuations.
class ThreadLocalState {
 public:
  ThreadLocalState(bool grad_mode_enabled, int64_t dist_autograd_context_id);

  static ThreadLocalState getThreadLocalState();

  static void setThreadLocalState(const ThreadLocalState& state);

 private:
  bool grad_mode_enabled_;
#ifdef USE_DISTRIBUTED
  int64_t dist_autograd_context_id_;
#endif
};

// Guard to set and reset the appropriate thread local state for JIT
// interpreter continuations.
class ThreadLocalStateGuard {
 public:
  explicit ThreadLocalStateGuard(const ThreadLocalState& state)
      : prev_state_(ThreadLocalState::getThreadLocalState()) {
    ThreadLocalState::setThreadLocalState(state);
  }

  ~ThreadLocalStateGuard() {
    ThreadLocalState::setThreadLocalState(prev_state_);
  }

 private:
  ThreadLocalState prev_state_;
};

} // namespace torch
