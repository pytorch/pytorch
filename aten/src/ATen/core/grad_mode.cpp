#include <ATen/core/grad_mode.h>

namespace at {

/// In the CAFFE2_FB_LIMITED_MOBILE_CAPABILITY build setting,
/// thread_local is not supported. In that case, we don't provide
/// `at::GradMode`.
#ifndef CAFFE2_FB_LIMITED_MOBILE_CAPABILITY

thread_local bool GradMode_enabled = true;

bool GradMode::is_enabled() {
  return GradMode_enabled;
}

void GradMode::set_enabled(bool enabled) {
  GradMode_enabled = enabled;
}

#else // defined(CAFFE2_FB_LIMITED_MOBILE_CAPABILITY)

bool GradMode::is_enabled() {
  throw std::runtime_error("GradMode is not supported on mobile");
}

void GradMode::set_enabled(bool enabled) {
  throw std::runtime_error("GradMode is not supported on mobile");
}

#endif

} // namespace at
