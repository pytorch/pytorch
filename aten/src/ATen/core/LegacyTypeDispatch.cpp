#include <ATen/core/LegacyTypeDispatch.h>

namespace at {

// TODO: This could be bad juju if someone calls globalContext() in the
// destructor of an object with static lifetime.
LegacyTypeDispatch & globalLegacyTypeDispatch() {
  static LegacyTypeDispatch singleton;
  return singleton;
}

}
