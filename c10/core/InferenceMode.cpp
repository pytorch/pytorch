#include <c10/core/InferenceMode.h>
#include <stdexcept>

namespace c10 {
thread_local bool InferenceMode_enabled = false;

// Invariant:
//   is_enabled() == !c10::impl::tls_is_dispatch_key_included(DispatchKey::InplaceOrView);
// InferenceMode::is_enabled() is in perf critical path (TensorImpl constructor)
// so it worths a separate TLS to skip the DispatchKeySet check.
bool InferenceMode::is_enabled() {
  return InferenceMode_enabled;
}

void InferenceMode::set_enabled(bool enabled) {
  InferenceMode_enabled = enabled;
}
} // namespace c10
