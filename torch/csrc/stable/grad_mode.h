#pragma once

#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/version.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/shim_utils.h>

HIDDEN_NAMESPACE_BEGIN(torch, stable)

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0

// An ABI-stable counterpart to c10::InferenceMode. The guard must be destroyed
// on the thread where it was constructed, in reverse order of construction.
// Minimum compatible version: PyTorch 2.16.
class InferenceMode {
 public:
  InferenceMode(bool enabled = true) {
    STABLE_TORCH_ERROR_CODE_CHECK(
        torch_new_inference_mode_guard(enabled, &guard_));
  }

  InferenceMode(const InferenceMode&) = delete;
  InferenceMode(InferenceMode&&) = delete;
  InferenceMode& operator=(const InferenceMode&) = delete;
  InferenceMode& operator=(InferenceMode&&) = delete;

  ~InferenceMode() {
    torch_delete_inference_mode_guard(guard_);
  }

  static bool is_enabled() {
    return torch_is_inference_mode_enabled();
  }

 private:
  InferenceModeGuardHandle guard_ = nullptr;
};

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0

HIDDEN_NAMESPACE_END(torch, stable)
