#include <torch/csrc/autograd/grad_mode.h>
#ifdef USE_DISTRIBUTED
#include <torch/csrc/distributed/autograd/context/container.h>
#endif
#include <torch/csrc/ThreadLocalState.h>

namespace torch {

#ifdef USE_DISTRIBUTED
using torch::distributed::autograd::DistAutogradContainer;
#endif

ThreadLocalState::ThreadLocalState(
    bool grad_mode_enabled,
    int64_t dist_autograd_context_id)
    : grad_mode_enabled_(grad_mode_enabled),
      dist_autograd_context_id_(dist_autograd_context_id) {}

bool ThreadLocalState::gradModeEnabled() const {
  return grad_mode_enabled_;
}

int64_t ThreadLocalState::distAutogradContextId() const {
  return dist_autograd_context_id_;
}

ThreadLocalState getThreadLocalState() {
  int64_t dist_autograd_context_id = -1;
#ifdef USE_DISTRIBUTED
  dist_autograd_context_id = DistAutogradContainer::currentContextId();
#endif

  return ThreadLocalState(
      autograd::GradMode::is_enabled(), dist_autograd_context_id);
}

void setThreadLocalState(const ThreadLocalState& state) {
  at::GradMode::set_enabled(state.gradModeEnabled());

#ifdef USE_DISTRIBUTED
  DistAutogradContainer::forceCurrentContextId(state.distAutogradContextId());
#endif
}

} // namespace torch
