#pragma once

#include <c10/macros/Macros.h>

namespace c10 {

// RAII thread local guard that tracks whether code is being executed in
// `at::parallel_for` or `at::parallel_reduce` loop function.
class ParallelGuard {
 private:
  static bool& get_in_at_parallel() {
    thread_local static bool in_at_parallel = false;
    return in_at_parallel;
  }

 public:
  static inline bool is_enabled() {
    return get_in_at_parallel();
  }

  inline ParallelGuard(bool state) : previous_state_(is_enabled()) {
    get_in_at_parallel() = state;
  }

  inline ~ParallelGuard() {
    get_in_at_parallel() = previous_state_;
  }

 private:
  bool previous_state_;
};

} // namespace c10
