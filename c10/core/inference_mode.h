#pragma once

#include <c10/macros/Macros.h>
#include <c10/core/impl/LocalDispatchKeySet.h>

namespace c10 {

struct TORCH_API InferenceMode {
  static bool is_enabled();
  static void set_enabled(bool enabled);
};

// A RAII, thread local (!) guard that enables or disables grad mode upon
// construction, and sets it back to the original value upon destruction.
struct TORCH_API AutoInferenceMode {
  // FIXME: remember prev_exclude_set
  AutoInferenceMode(bool enabled) : prev_mode(InferenceMode::is_enabled()), 
    autograd_guard_(enabled? c10::autograd_dispatch_keyset : DispatchKeySet())  {
    InferenceMode::set_enabled(enabled);
  }
  ~AutoInferenceMode() {
    InferenceMode::set_enabled(prev_mode);
  }
  bool prev_mode;
  // disable all autograd dispatch keys
  c10::impl::ExcludeDispatchKeyGuard autograd_guard_;
};

}
