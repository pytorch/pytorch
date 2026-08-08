#include <c10/core/AutogradState.h>

#include <utility>

namespace c10 {

namespace {
// By default, grad mode is enabled, inference mode and multithreading are
// disabled,
thread_local AutogradState autograd_state_tls = AutogradState(
    /* grad_mode */ true,
    /* inference_mode */ false,
    /* fw_grad_mode */ true,
    /* multithreading_enabled */ false);
} // namespace

AutogradState& AutogradState::get_tls_state() {
  return autograd_state_tls;
}

void AutogradState::set_tls_state(AutogradState state) {
  autograd_state_tls = std::move(state);
}

} // namespace c10
