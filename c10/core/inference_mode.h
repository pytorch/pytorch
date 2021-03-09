#pragma once

#include <c10/macros/Macros.h>
#include <c10/core/impl/LocalDispatchKeySet.h>

namespace c10 {

struct TORCH_API InferenceMode {
  static bool is_enabled();
  static void set_enabled(bool enabled);
};

// A RAII, thread local (!) guard that enables or disables inference mode upon
// construction, and sets it back to the original value upon destruction.
struct TORCH_API AutoInferenceMode {
  AutoInferenceMode(bool enabled) : prev_mode(InferenceMode::is_enabled()),
  autograd_guard_(enabled? c10::autograd_dispatch_keyset : DispatchKeySet()) {
    InferenceMode::set_enabled(enabled);
  }
  ~AutoInferenceMode() {
    InferenceMode::set_enabled(prev_mode);
  }
  bool prev_mode;
  // When AutoInferenceMode is enabled, Autograd dispatch keys are excluded
  // but not InplaceOrView key.
  //
  // For example:
  //  torch::Tensor a = torch::ones({1, 2, 3}).set_requires_grad(true);
  //  torch::Tensor k = a + 2;
  //  {
  //    c10::AutoInferenceMode guard(true);
  //    k.add_(2);
  //  }
  //  `k.add_(2)` still need to go through InplaceOrView kernel so that it's
  //  prepared for future autograd.
  c10::impl::ExcludeDispatchKeyGuard autograd_guard_;
};
} // namespace c10

