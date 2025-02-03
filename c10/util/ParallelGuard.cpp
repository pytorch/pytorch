#include <c10/util/ParallelGuard.h>

namespace c10 {

thread_local static bool in_at_parallel = false;

bool ParallelGuard::is_enabled() {
  return in_at_parallel;
}

ParallelGuard::ParallelGuard(bool state) : previous_state_(is_enabled()) {
  in_at_parallel = state;
}

ParallelGuard::~ParallelGuard() {
  in_at_parallel = previous_state_;
}

} // namespace c10
