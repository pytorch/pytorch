#include <ATen/LegacyTHDispatch.h>

namespace at {

// TODO: This could be bad juju if someone calls globalContext() in the
// destructor of an object with static lifetime.
LegacyTHDispatch & globalLegacyTHDispatch() {
  static LegacyTHDispatch singleton;
  return singleton;
}

}
