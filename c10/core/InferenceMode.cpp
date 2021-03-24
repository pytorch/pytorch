
#include <c10/core/InferenceMode.h>

#include <stdexcept>

namespace c10 {

bool InferenceMode::is_enabled() {
  return !c10::impl::tls_is_dispatch_key_included(DispatchKey::InplaceOrView);
}

} // namespace c10
