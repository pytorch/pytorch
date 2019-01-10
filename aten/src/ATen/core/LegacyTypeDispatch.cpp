#include <ATen/core/LegacyTypeDispatch.h>

namespace at {

thread_local bool NonVariableTypeMode_enabled = false;

bool NonVariableTypeMode::is_enabled() {
  return NonVariableTypeMode_enabled;
}

void NonVariableTypeMode::set_enabled(bool enabled) {
  NonVariableTypeMode_enabled = enabled;
}

// TODO: This could be bad juju if someone calls globalContext() in the
// destructor of an object with static lifetime.
LegacyTypeDispatch & globalLegacyTypeDispatch() {
  static LegacyTypeDispatch singleton;
  return singleton;
}

}
