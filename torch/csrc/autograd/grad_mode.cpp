#include "grad_mode.h"

namespace torch { namespace autograd {

thread_local bool GradMode_enabled = true;

bool GradMode::is_enabled() noexcept {
  return GradMode_enabled;
}

void GradMode::set_enabled(bool enabled) noexcept {
  GradMode_enabled = enabled;
}
}}
