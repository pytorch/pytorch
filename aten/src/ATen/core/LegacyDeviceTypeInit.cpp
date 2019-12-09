#include <ATen/core/LegacyDeviceTypeInit.h>

namespace at {

C10_DEFINE_REGISTRY(
    LegacyDeviceTypeInitRegistry,
    LegacyDeviceTypeInitInterface,
    LegacyDeviceTypeInitArgs)

const LegacyDeviceTypeInitInterface& getLegacyDeviceTypeInit() {
  static std::unique_ptr<LegacyDeviceTypeInitInterface> legacy_device_type_init;
  static std::once_flag once;
  std::call_once(once, [] {
    legacy_device_type_init = LegacyDeviceTypeInitRegistry()->Create("LegacyDeviceTypeInit", LegacyDeviceTypeInitArgs{});
    if (!legacy_device_type_init) {
      legacy_device_type_init =
          std::unique_ptr<LegacyDeviceTypeInitInterface>(new LegacyDeviceTypeInitInterface());
    }
  });
  return *legacy_device_type_init;
}

}
