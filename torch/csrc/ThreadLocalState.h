#pragma once

namespace torch {

// Thread local state that needs to be propagated across different threads
// during JIT interpreter continuations.
class ThreadLocalState {
 public:
  ThreadLocalState(bool grad_mode_enabled, int64_t dist_autograd_context_id);
  bool gradModeEnabled() const;
  int64_t distAutogradContextId() const;

 private:
  bool grad_mode_enabled_;
  int64_t dist_autograd_context_id_;
};

ThreadLocalState getThreadLocalState();

void setThreadLocalState(const ThreadLocalState& state);

// Guard to set and reset the appropriate thread local state for JIT
// interpreter continuations.
class ThreadLocalStateGuard {
 public:
  explicit ThreadLocalStateGuard(const ThreadLocalState& state)
      : prev_state_(getThreadLocalState()) {
    setThreadLocalState(state);
  }

  ~ThreadLocalStateGuard() {
    setThreadLocalState(prev_state_);
  }

 private:
  ThreadLocalState prev_state_;
};

} // namespace torch
