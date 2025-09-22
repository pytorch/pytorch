#include <c10/core/impl/HermeticPyObjectTLS.h>

namespace c10::impl {

// In single-interpreter mode, these are simplified stubs

void HermeticPyObjectTLS::set_state(bool state) {
  // No-op in single-interpreter mode
  // Kept for backward compatibility
}

bool HermeticPyObjectTLS::get_tls_state() {
  // Always return false in single-interpreter mode
  return false;
}

} // namespace c10::impl
