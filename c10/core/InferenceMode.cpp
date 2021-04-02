
#include <c10/core/InferenceMode.h>

namespace c10 {

bool InferenceMode::is_enabled() {
  // See Note [Expected TLS state in InferenceMode]
  return !c10::impl::tls_is_dispatch_key_included(DispatchKey::InplaceOrView);
}

} // namespace c10
