#pragma once

#include <c10/macros/Macros.h>

namespace c10 {

// RAII thread local guard that tracks whether code is being executed in
// `at::parallel_for` or `at::parallel_reduce` loop function.
class C10_API ParallelGuard {
 public:
  static bool is_enabled();

  ParallelGuard(bool state);
  ~ParallelGuard();

 private:
  bool previous_state_;
};

} // namespace c10
