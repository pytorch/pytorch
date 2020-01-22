#include <ATen/native/mobile/cpu/Engine.h>

#ifdef USE_XNNPACK

#include <ATen/native/mobile/cpu/internal/Common.h>

namespace at {
namespace native {
namespace mobile {
namespace cpu {
namespace internal {
namespace {

bool is_initialized_ = false;

} // namespace internal
} // namespace

bool available() {
  // Add extra conditions here that should disable mobile CPU impl at runtime in its totality.
  return initialize();
}

bool initialize() {
  using namespace internal;

  // This implementation allows for retries.
  if (!is_initialized_) {
    const xnn_status status = xnn_initialize(nullptr);
    is_initialized_ = (xnn_status_success == status);

    if (!is_initialized_) {
      // Avoid polluting the logs.
      static bool failure_logged_ = false;

      if (!failure_logged_) {
        if (xnn_status_out_of_memory == status) {
          LOG(WARNING) << "Failed to initialize XNNPACK! Reason: Out of memory.";
        } else if (xnn_status_unsupported_hardware == status) {
          LOG(WARNING) << "Failed to initialize XNNPACK! Reason: Unsupported hardware.";
        } else {
          LOG(WARNING) << "Failed to initialize XNNPACK! Reason: Unknown error!";
        }

        failure_logged_ = true;
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
      // Avoid polluting the logs.
      static bool failure_logged_ = false;

      if (!failure_logged_) {
        LOG(WARNING) << "Failed to uninitialize XNNPACK! Reason: Unknown error!";

        failure_logged_ = true;
      }
    }
  }

  return !is_initialized_;
}

} // namespace cpu
} // namespace mobile
} // namespace native
} // namespace at

#endif /* USE_XNNPACK */
