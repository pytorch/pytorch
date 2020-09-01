#include <ATen/VmapMode.h>

namespace at {
namespace impl {

/// thread_local is a feature that is not enabled by Caffe2 mobile
/// build (e.g. iOS). Therefore, we only provide `at::VmapMode`
/// when we are not in mobile build or when FEATURE_TORCH_MOBILE
/// is on.
#if !defined(C10_MOBILE) || defined(FEATURE_TORCH_MOBILE)

thread_local int64_t VmapMode_current_vmap_level = 0;

int64_t VmapMode::current_vmap_level() {
  return VmapMode_current_vmap_level;
}

int64_t VmapMode::increment_nesting() {
  VmapMode_current_vmap_level++;
  if (VmapMode_current_vmap_level == 1) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::VmapMode, true);
  }
  return VmapMode_current_vmap_level;
}

int64_t VmapMode::decrement_nesting() {
  VmapMode_current_vmap_level--;
  if (VmapMode_current_vmap_level == 0) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::VmapMode, false);
  }
  return VmapMode_current_vmap_level;
}

#else

int64_t VmapMode::current_nesting_level() {
  TORCH_CHECK(false, "VmapMode is not supported on mobile");
}

int64_t VmapMode::increment_nesting() {
  TORCH_CHECK(false, "VmapMode is not supported on mobile");
}

int64_t VmapMode::decrement_nesting() {
  TORCH_CHECK(false, "VmapMode is not supported on mobile");
}

#endif

} // namespace impl
} // namespace at
