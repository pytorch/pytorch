#include <c10/core/AutogradState.h>

namespace c10 {

namespace {

// By default, grad mode and multithreading are enabled, inference mode is
// disabled,
// NB: MSVC has issues with constinit thread_local, so only use it on other
// compilers.
#if defined(_MSC_VER)
thread_local AutogradState autograd_state_tls = AutogradState(
#else
constinit thread_local AutogradState autograd_state_tls = AutogradState(
#endif
    /* grad_mode */ true,
    /* inference_mode */ false,
    /* fw_grad_mode */ true,
    /* multithreading_enabled */ true);
} // namespace

AutogradState& AutogradState::get_tls_state() {
  return autograd_state_tls;
}

void AutogradState::set_tls_state(AutogradState state) {
  autograd_state_tls = state;
}

} // namespace c10
