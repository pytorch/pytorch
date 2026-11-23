#pragma once

#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/version.h>
#include <torch/headeronly/macros/Macros.h>

HIDDEN_NAMESPACE_BEGIN(torch, stable)

// ABI-stable counterparts to the guards in c10/core/GradMode.h. Guards must be
// destroyed on the thread where they were constructed, in reverse order.
// Minimum compatible version: PyTorch 2.9.
// Build time minimum version: PyTorch 2.16.
struct GradMode {
  static bool is_enabled() {
    return aoti_torch_grad_mode_is_enabled();
  }

  static void set_enabled(bool enabled) {
    aoti_torch_grad_mode_set_enabled(enabled);
  }
};

struct AutoGradMode {
  AutoGradMode(bool enabled) : prev_mode(GradMode::is_enabled()) {
    GradMode::set_enabled(enabled);
  }
  AutoGradMode(const AutoGradMode&) = delete;
  AutoGradMode(AutoGradMode&&) = delete;
  AutoGradMode& operator=(const AutoGradMode&) = delete;
  AutoGradMode& operator=(AutoGradMode&&) = delete;
  ~AutoGradMode() {
    GradMode::set_enabled(prev_mode);
  }
  bool prev_mode;
};

struct NoGradGuard : public AutoGradMode {
  NoGradGuard() : AutoGradMode(/*enabled=*/false) {}
};

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0

// Minimum compatible version: PyTorch 2.16
// Build time minimum version: PyTorch 2.16.
struct AutoFwGradMode {
  AutoFwGradMode(bool enabled) : prev_mode(torch_is_fw_grad_mode_enabled()) {
    torch_set_fw_grad_mode_enabled(enabled);
  }
  AutoFwGradMode(const AutoFwGradMode&) = delete;
  AutoFwGradMode(AutoFwGradMode&&) = delete;
  AutoFwGradMode& operator=(const AutoFwGradMode&) = delete;
  AutoFwGradMode& operator=(AutoFwGradMode&&) = delete;
  ~AutoFwGradMode() {
    torch_set_fw_grad_mode_enabled(prev_mode);
  }
  bool prev_mode;
};

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0

HIDDEN_NAMESPACE_END(torch, stable)
