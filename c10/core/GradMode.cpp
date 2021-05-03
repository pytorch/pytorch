#include <c10/core/GradMode.h>

#include <stdexcept>

namespace c10 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
thread_local bool GradMode_enabled = true;

bool GradMode::is_enabled() {
  return GradMode_enabled;
}

void GradMode::set_enabled(bool enabled) {
  GradMode_enabled = enabled;
}
} // namespace c10
