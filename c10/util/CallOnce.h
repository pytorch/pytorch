#pragma once

#include <atomic>
#include <mutex>
#include <thread>
#include <utility>

#include <c10/macros/Macros.h>
#include <c10/util/C++17.h>

namespace c10 {

// custom c10 call_once implementation to avoid the deadlock in std::call_once.
// The implementation here is a simplified version from folly and likely much
// much higher memory footprint.
template <typename Flag, typename F, typename... Args>
inline void call_once(Flag& flag, F&& f, Args&&... args) {
  if (C10_LIKELY(flag.test_once())) {
    return;
  }
  flag.call_once_slow(std::forward<F>(f), std::forward<Args>(args)...);
}

class once_flag {
 public:
  constexpr once_flag() noexcept = default;
  once_flag(const once_flag&) = delete;
  once_flag& operator=(const once_flag&) = delete;

 private:
  template <typename Flag, typename F, typename... Args>
  friend void call_once(Flag& flag, F&& f, Args&&... args);

  template <typename F, typename... Args>
  void call_once_slow(F&& f, Args&&... args) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (init_.load(std::memory_order_relaxed)) {
      return;
    }
    c10::guts::invoke(f, std::forward<Args>(args)...);
    init_.store(true, std::memory_order_release);
  }

  bool test_once() {
    return init_.load(std::memory_order_acquire);
  }

  void reset_once() {
    init_.store(false, std::memory_order_release);
  }

 private:
  std::mutex mutex_;
  std::atomic<bool> init_{false};
};

} // namespace c10
