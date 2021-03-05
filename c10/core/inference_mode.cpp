
#include <c10/core/inference_mode.h>

#include <stdexcept>

namespace c10 {

/// thread_local is a feature that is not enabled by Caffe2 mobile
/// build (e.g. iOS). Therefore, we only provide `at::InferenceMode`
/// when we are not in mobile build or when FEATURE_TORCH_MOBILE
/// is on.
#if !defined(C10_MOBILE) || defined(FEATURE_TORCH_MOBILE)

thread_local bool InferenceMode_enabled = false;

bool InferenceMode::is_enabled() {
  return InferenceMode_enabled;
}

void InferenceMode::set_enabled(bool enabled) {
  InferenceMode_enabled = enabled;
  // FIXME: link to test
  // See quip error handling (1) (3) (6)
  c10::impl::tls_set_dispatch_key_included(DispatchKey::InplaceOrView, !enabled);
}

#else

bool InferenceMode::is_enabled() {
  return false;
}

void InferenceMode::set_enabled(bool enabled) {
  throw std::runtime_error("InferenceMode::set_enabled is not supported on mobile");
}

#endif

} // namespace c10
