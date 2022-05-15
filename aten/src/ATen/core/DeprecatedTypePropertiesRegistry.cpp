#include <ATen/core/DeprecatedTypePropertiesRegistry.h>

#include <ATen/core/DeprecatedTypeProperties.h>
#include <c10/util/irange.h>

namespace at {

void DeprecatedTypePropertiesDeleter::operator()(DeprecatedTypeProperties * ptr) {
  delete ptr;
}

DeprecatedTypePropertiesRegistry::DeprecatedTypePropertiesRegistry() {
  for (const auto b : c10::irange(static_cast<int>(Backend::NumOptions))) {
    for (const auto s : c10::irange(static_cast<int>(ScalarType::NumOptions))) {
      registry[b][s] = std::make_unique<DeprecatedTypeProperties>(
              static_cast<Backend>(b),
              static_cast<ScalarType>(s));
    }
  }
}

DeprecatedTypeProperties& DeprecatedTypePropertiesRegistry::getDeprecatedTypeProperties(
    Backend p, ScalarType s) const {
  return *registry[static_cast<int>(p)][static_cast<int>(s)];
}

// TODO: This could be bad juju if someone calls globalContext() in the
// destructor of an object with static lifetime.
DeprecatedTypePropertiesRegistry & globalDeprecatedTypePropertiesRegistry() {
  static DeprecatedTypePropertiesRegistry singleton;
  return singleton;
}

}
