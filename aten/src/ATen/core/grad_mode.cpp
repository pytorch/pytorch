#include <ATen/core/grad_mode.h>

#include <stdexcept>

namespace at {

thread_local bool GradMode_enabled = true;

bool GradMode::is_enabled() {
  return GradMode_enabled;
}

void GradMode::set_enabled(bool enabled) {
  GradMode_enabled = enabled;
}
} // namespace at
