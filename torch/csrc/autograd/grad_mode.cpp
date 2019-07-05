#include <torch/csrc/autograd/grad_mode.h>

namespace torch { namespace autograd {

#ifndef C10_MOBILE

thread_local bool GradMode_enabled = true;

bool GradMode::is_enabled() {
  return GradMode_enabled;
}

void GradMode::set_enabled(bool enabled) {
  GradMode_enabled = enabled;
}

#else // defined(C10_MOBILE)

bool GradMode::is_enabled() {
  throw std::runtime_error("GradMode is not supported on mobile");
}

void GradMode::set_enabled(bool enabled) {
  throw std::runtime_error("GradMode is not supported on mobile");
}

#endif

}}
