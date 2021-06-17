#include <c10/core/GradMode.h>

namespace c10 {
bool GradMode::is_enabled() {
  return is_enabled(impl::_get_thread_local_state());
}

void GradMode::set_enabled(bool enabled) {
  impl::_get_thread_local_state()->GradMode_disabled = !enabled;
}
} // namespace c10
