#include <ATen/ATen.h>
#include <ATen/core/autocast/autocast_mode.h>
#include <c10/core/impl/LocalTensorTypeSet.h>

#include <stdexcept>
#include <memory>
#include <unordered_map>

namespace at {
namespace autocast {

/// thread_local is a feature that is not enabled by Caffe2 mobile
/// build (e.g. iOS). Therefore, we only provide `at::AutocastMode`
/// when we are not in mobile build or when FEATURE_TORCH_MOBILE
/// is on.
#if !defined(C10_MOBILE) || defined(FEATURE_TORCH_MOBILE)
namespace {
  thread_local std::unique_ptr<c10::impl::IncludeTensorTypeIdGuard> guard_holder;
}

bool AutocastMode::is_enabled() {
  return bool(guard_holder.get());
}

void AutocastMode::set_enabled(bool enabled) {
  if (enabled) {
    if (!is_enabled()) {
      guard_holder.reset(new c10::impl::IncludeTensorTypeIdGuard(TensorTypeId::AutocastTensorId));
    }
  } else {
    guard_holder.reset(nullptr);
  }
}

// I'd like the cache to reside in the same .cpp file that actually does the casting, for quicker access.
// Clearing the cache is something that happens relatively rarely.
std::unordered_map<TensorImpl*, Tensor> & get_cache();

void AutocastMode::clear_cache() {
  at::autocast::get_cache().clear();
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

#endif

} // namespace autocast
} // namespace at
