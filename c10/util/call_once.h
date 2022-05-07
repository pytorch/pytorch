#pragma once

#include <mutex>

namespace c10 {

struct once_flag {
  std::mutex mutex_;
  bool has_run_ = false;
};

// std::call_once has a bug in exception handling
template <class Callable, class... Args>
void call_once(once_flag& flag, Callable&& f, Args&&... args) {
  std::lock_guard<std::mutex> lock(flag.mutex_);
  if (flag.has_run_) {
    return;
  }
  f(std::forward<Args>(args)...);
  flag.has_run_ = true;
}

} // namespace c10
