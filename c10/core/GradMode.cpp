#include <c10/core/GradMode.h>
#include <c10/core/AutogradMode.h>

#include <stdexcept>

namespace c10 {

bool GradMode::is_enabled() {
  return AutogradMode::get_grad_mode();
}

void GradMode::set_enabled(bool enabled) {
  AutogradMode::set_grad_mode(enabled);
}
} // namespace c10
