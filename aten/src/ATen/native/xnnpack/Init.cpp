#ifdef USE_XNNPACK

#include <ATen/native/xnnpack/Common.h>

namespace at {
namespace native {
namespace xnnpack {
namespace internal {
namespace {

bool is_initialized_ = false;

bool initialize() {
  using namespace internal;

  // This implementation allows for retries.
  if (!is_initialized_) {
    const xnn_status status = xnn_initialize(nullptr);
    is_initialized_ = (xnn_status_success == status);

    if (!is_initialized_) {
      if (xnn_status_out_of_memory == status) {
        TORCH_WARN_ONCE("Failed to initialize XNNPACK! Reason: Out of memory.");
      } else if (xnn_status_unsupported_hardware == status) {
        TORCH_WARN_ONCE("Failed to initialize XNNPACK! Reason: Unsupported hardware.");
      } else {
        TORCH_WARN_ONCE("Failed to initialize XNNPACK! Reason: Unknown error!");
      }
    }
  }

  return is_initialized_;
}

bool deinitialize() {
  using namespace internal;

  // This implementation allows for retries.
  if (is_initialized_) {
    const xnn_status status = xnn_deinitialize();
    is_initialized_ = !(xnn_status_success == status);

    if (is_initialized_) {
      TORCH_WARN_ONCE("Failed to uninitialize XNNPACK! Reason: Unknown error!");
    }
  }

  return !is_initialized_;
}

} // namespace

bool available() {
  // Add extra conditions here that should disable mobile CPU impl at runtime in its totality.
  return internal::initialize();
}

} // namespace internal
} // namespace xnnpack
} // namespace native
} // namespace at

#endif /* USE_XNNPACK */
