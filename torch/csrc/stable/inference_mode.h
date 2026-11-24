#pragma once

#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/version.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/shim_utils.h>

HIDDEN_NAMESPACE_BEGIN(torch, stable)

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0

/**
 * @brief ABI-stable counterpart to c10::InferenceMode in
 * c10/core/InferenceMode.h.
 *
 * Preserves the thread-local RAII behavior of c10::InferenceMode. Destroy
 * guards on their creating thread, in reverse construction order.
 *
 * The stable wrapper owns a heap-allocated guard through an opaque handle.
 * Construction can throw if allocation fails.
 *
 * Minimum compatible version: PyTorch 2.16.
 * Build time minimum version: PyTorch 2.16.
 */
class InferenceMode {
 public:
  /// Sets inference mode on the current thread.
  InferenceMode(bool enabled = true) {
    STABLE_TORCH_ERROR_CODE_CHECK(
        torch_create_inference_mode_guard(enabled, &guard_));
  }

  InferenceMode(const InferenceMode&) = delete;
  InferenceMode(InferenceMode&&) = delete;
  InferenceMode& operator=(const InferenceMode&) = delete;
  InferenceMode& operator=(InferenceMode&&) = delete;

  /// Restores the previous thread-local state.
  ~InferenceMode() {
    // The underlying guard destruction is nonthrowing.
    (void)torch_delete_inference_mode_guard(guard_);
  }

  /// Returns whether inference mode is enabled on the current thread.
  static bool is_enabled() {
    return torch_is_inference_mode_enabled();
  }

 private:
  InferenceModeGuardHandle guard_ = nullptr;
};

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0

HIDDEN_NAMESPACE_END(torch, stable)
