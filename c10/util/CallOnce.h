#pragma once

#include <atomic>
#include <mutex>
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
#ifndef _WIN32
  // running into build error on MSVC. Can't seem to get a repro locally so I'm
  // just avoiding constexpr
  //
  //   C:/actions-runner/_work/pytorch/pytorch\c10/util/CallOnce.h(26): error:
  //   defaulted default constructor cannot be constexpr because the
  //   corresponding implicitly declared default constructor would not be
  //   constexpr 1 error detected in the compilation of
  //   "C:/actions-runner/_work/pytorch/pytorch/aten/src/ATen/cuda/cub.cu".
  constexpr
#endif
      once_flag() noexcept = default;
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
