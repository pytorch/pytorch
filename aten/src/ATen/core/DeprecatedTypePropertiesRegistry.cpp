#include <ATen/core/DeprecatedTypePropertiesRegistry.h>

namespace at {

// TODO: This could be bad juju if someone calls globalContext() in the
// destructor of an object with static lifetime.
DeprecatedTypePropertiesRegistry & globalDeprecatedTypePropertiesRegistry() {
  static DeprecatedTypePropertiesRegistry singleton;
  return singleton;
}

}
