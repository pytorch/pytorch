#pragma once

#include <c10/macros/Macros.h>

namespace c10 {

// Returns true if executed while a ParallelGuard object is alive in the current
// thread.
C10_API bool is_parallel_guard_alive();

class C10_API ParallelGuard {
 public:
  ParallelGuard(bool state);
  ~ParallelGuard();

 private:
  bool previous_state_;
};

} // namespace c10
