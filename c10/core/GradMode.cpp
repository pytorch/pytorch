#include <c10/core/GradMode.h>

#include <stdexcept>

namespace c10 {

bool GradMode::is_enabled() {
  return AutogradTLS::get_grad_mode();
}

void GradMode::set_enabled(bool enabled) {
  AutogradTLS::set_grad_mode(enabled);
}
} // namespace c10
