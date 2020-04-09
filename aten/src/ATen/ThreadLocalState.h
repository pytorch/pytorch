#pragma once

#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/util/Exception.h>

#include <ATen/ThreadLocalDebugInfo.h>

namespace at {

// Enumerating some of the thread local
// settings we're interested in propagating
// across threads and network
enum class ThreadLocalSetting : uint8_t {
  // Note: grad_mode is not automatically propagated from
  // the forward pass into the backward pass threads
  GRAD_MODE = 0,
  // Whether or not to execute record functions
  RECORD_FUNCTION,
  // Profiler enabled/disabled
  PROFILER,
  NUM_SETTINGS, // must be the last in the list
};

// A simple representation of a thread local value
union SettingValue {
  int64_t value;
  struct {
    int32_t first;
    int32_t second;
  } pair;
};

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

  // Some of the settings (e.g. profiler) may register
  // getters and setters,
  // WARNING: not thread safe
  static void registerThreadLocalSetting(
      ThreadLocalSetting st,
      std::function<SettingValue(void)> getter,
      std::function<void(SettingValue)> setter);

 private:
  c10::impl::LocalDispatchKeySet dispatch_key_;

  // ThreadLocalDebugInfo does not change after being created
  // with DebugInfoGuard
  std::shared_ptr<at::ThreadLocalDebugInfo> debug_info_;
  bool keep_grad_mode_ = true;

  std::array<SettingValue, (size_t)ThreadLocalSetting::NUM_SETTINGS> settings_;

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

// Internal, turns on/off record function observers
TORCH_API bool _tls_is_record_function_enabled();
TORCH_API void _tls_set_record_function_enabled(bool);

} // namespace at
