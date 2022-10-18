#include <c10/core/impl/HermeticPyObjectTLS.h>

namespace c10 {
namespace impl {

// If this TLS access is a bottleneck, you can optimize it by introducing
// a global variable that is tested first, before we check TLS.  That global
// variable would be set to true when we launch a multipy/torchdeploy
// interpreter.
thread_local bool hermeticPyObjectState = false;

bool HermeticPyObjectTLS::haveState_{false};

void HermeticPyObjectTLS::set_state(bool state) {
  hermeticPyObjectState = state;
}

bool HermeticPyObjectTLS::get_tls_state() {
  return hermeticPyObjectState;
}

void HermeticPyObjectTLS::init_state() {
  haveState_ = true;
}

} // namespace impl
} // namespace c10
