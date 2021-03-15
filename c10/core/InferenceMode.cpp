
#include <c10/core/InferenceMode.h>

#include <stdexcept>

namespace c10 {

// Invariant:
//   Normal mode: InplaceOrView in TLS included
//   InferenceMode: InferenceMode not in TLS included
bool InferenceMode::is_enabled() {
  return !c10::impl::tls_is_dispatch_key_included(DispatchKey::InplaceOrView);
}

} // namespace c10
