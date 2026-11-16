#include <c10/core/GradMode.h>

namespace c10 {

bool GradMode::is_enabled() {
  return AutogradState::get_tls_state().get_grad_mode();
}

void GradMode::set_enabled(bool enabled) {
  AutogradState::get_tls_state().set_grad_mode(enabled);
}
} // namespace c10
