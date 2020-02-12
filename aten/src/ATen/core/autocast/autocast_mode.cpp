#include <ATen/ATen.h>
#include <ATen/core/autocast/autocast_mode.h>
#include <c10/core/impl/LocalDispatchKeySet.h>

#include <stdexcept>
#include <memory>

namespace at {
namespace autocast {

/// thread_local is a feature that is not enabled by Caffe2 mobile
/// build (e.g. iOS). Therefore, we only provide `at::AutocastMode`
/// when we are not in mobile build or when FEATURE_TORCH_MOBILE
/// is on.
#if !defined(C10_MOBILE) || defined(FEATURE_TORCH_MOBILE)
bool AutocastMode::is_enabled() {
  return c10::impl::tls_is_dispatch_key_included(DispatchKey::AutocastTensorId);
}

void AutocastMode::set_enabled(bool new_enabled) {
  c10::impl::tls_set_dispatch_key_included(DispatchKey::AutocastTensorId, new_enabled);
}

void clear_cache();
int increment_nesting();
int decrement_nesting();

void AutocastMode::clear_cache() {
  at::autocast::clear_cache();
}

int AutocastMode::increment_nesting() {
  at::autocast::increment_nesting();
}

int AutocastMode::decrement_nesting() {
  at::autocast::decrement_nesting();
}

#else

bool AutocastMode::is_enabled() {
  throw std::runtime_error("AutocastMode is not supported on mobile");
}

void AutocastMode::set_enabled(bool enabled) {
  throw std::runtime_error("Autocast is not supported on mobile");
}

void AutocastMode::clear_cache() {
  throw std::runtime_error("Autocast is not supported on mobile");
}

int AutocastMode::increment_nesting() {
  throw std::runtime_error("Autocast is not supported on mobile");
}

int AutocastMode::decrement_nesting() {
  throw std::runtime_error("Autocast is not supported on mobile");
}

#endif

} // namespace autocast
} // namespace at
