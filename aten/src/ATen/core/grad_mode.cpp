#include <ATen/core/grad_mode.h>

namespace at {

/// thread_local is a feature that is not enabled by caffe2 mobile
/// build (e.g. iOS). Therefore, we only provide `at::GradMode`
/// when FEATURE_TORCH_MOBILE is on.
#ifdef FEATURE_TORCH_MOBILE

thread_local bool GradMode_enabled = true;

bool GradMode::is_enabled() {
  return GradMode_enabled;
}

void GradMode::set_enabled(bool enabled) {
  GradMode_enabled = enabled;
}

#else // defined(FEATURE_TORCH_MOBILE)

bool GradMode::is_enabled() {
  throw std::runtime_error("GradMode is not supported on mobile");
}

void GradMode::set_enabled(bool enabled) {
  throw std::runtime_error("GradMode is not supported on mobile");
}

#endif

} // namespace at
