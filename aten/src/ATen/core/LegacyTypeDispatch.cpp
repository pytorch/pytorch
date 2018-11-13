#include <ATen/core/LegacyTypeDispatch.h>

namespace at {

// TODO: This could be bad juju if someone calls globalContext() in the
// destructor of an object with static lifetime.
LegacyTypeDispatch & globalLegacyTypeDispatch() {
  static LegacyTypeDispatch singleton;
  return singleton;
}

C10_DEFINE_REGISTRY(
    LegacyTypeInitRegistry,
    LegacyTypeInitInterface,
    LegacyTypeInitArgs)

const LegacyTypeInitInterface& getLegacyTypeInit() {
  static std::unique_ptr<LegacyTypeInitInterface> legacy_type_init;
  static std::once_flag once;
  std::call_once(once, [] {
    legacy_type_init = LegacyTypeInitRegistry()->Create("LegacyTypeInit", LegacyTypeInitArgs{});
    if (!legacy_type_init) {
      legacy_type_init =
          std::unique_ptr<LegacyTypeInitInterface>(new LegacyTypeInitInterface());
    }
  });
  return *legacy_type_init;
}

}
