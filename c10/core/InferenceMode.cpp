#include <c10/core/InferenceMode.h>
#include <stdexcept>

namespace c10 {
/// thread_local is a feature that is not enabled by Caffe2 mobile
/// build (e.g. iOS). Therefore, we only provide `at::GradMode`
/// when we are not in mobile build or when FEATURE_TORCH_MOBILE
/// is on.
#if !defined(C10_MOBILE) || defined(FEATURE_TORCH_MOBILE)
thread_local bool InferenceMode_enabled = false;

// We could have skipped adding a new TLS InferenceMode_enabled
// by checking:
//   !c10::impl::tls_is_dispatch_key_included(DispatchKey::InplaceOrView);
// in InferenceMode::is_enabled().  But InferenceMode::is_enabled()
// is in perf critical path (TensorImpl constructor) so it worths
// a new TLS to skip the DispatchKeySet check.
bool InferenceMode::is_enabled() {
  return InferenceMode_enabled;
}

void InferenceMode::set_enabled(bool enabled) {
  InferenceMode_enabled = enabled;
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
