#include <ATen/core/grad_mode.h>

#include <stdexcept>

namespace at {

/// thread_local is a feature that is not enabled by Caffe2 mobile
/// build (e.g. iOS). Therefore, we only provide `at::GradMode`
/// when we are not in mobile build or when FEATURE_TORCH_MOBILE
/// is on.
#if !defined(C10_MOBILE) || defined(FEATURE_TORCH_MOBILE)

thread_local bool GradMode_enabled = true;

bool GradMode::is_enabled() {
  return GradMode_enabled;
}

void GradMode::set_enabled(bool enabled) {
  GradMode_enabled = enabled;
}

/// For now, forward grad is not broadly available so turn it
/// off by default to avoid leaking its state to user code
thread_local bool FwGradMode_enabled = false;

bool FwGradMode::is_enabled() {
  return FwGradMode_enabled;
}

void FwGradMode::set_enabled(bool enabled) {
  FwGradMode_enabled = enabled;
}

#else

bool GradMode::is_enabled() {
  return false;
}

void GradMode::set_enabled(bool enabled) {
  throw std::runtime_error("GradMode::set_enabled is not supported on mobile");
}

bool FwGradMode::is_enabled() {
  return false;
}

void FwGradMode::set_enabled(bool enabled) {
  throw std::runtime_error("FwGradMode::set_enabled is not supported on mobile");
}

#endif

} // namespace at
