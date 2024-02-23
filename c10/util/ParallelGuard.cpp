#include <c10/util/ParallelGuard.h>

namespace c10 {

thread_local bool in_at_parallel = false;

C10_API bool is_parallel_guard_alive() {
  return in_at_parallel;
}

ParallelGuard::ParallelGuard(bool state)
    : previous_state_(is_parallel_guard_alive()) {
  in_at_parallel = state;
}

ParallelGuard::~ParallelGuard() {
  in_at_parallel = previous_state_;
}

} // namespace c10
