#include <ATen/core/LegacyTypeDispatch.h>

namespace at {

// TODO: This could be bad juju if someone calls globalContext() in the
// destructor of an object with static lifetime.
LegacyTypeDispatch & globalLegacyTypeDispatch() {
  static LegacyTypeDispatch singleton;
  return singleton;
}

void LegacyTypeDispatch::initCPU() {
  static std::once_flag cpu_once;
  std::call_once(cpu_once, [] {
    getLegacyDeviceTypeInit().initCPU();
  });
}

void LegacyTypeDispatch::initCUDA() {
  static std::once_flag cuda_once;
  std::call_once(cuda_once, [] {
    getLegacyDeviceTypeInit().initCUDA();
  });
}

void LegacyTypeDispatch::initHIP() {
  static std::once_flag hip_once;
  std::call_once(hip_once, [] {
    getLegacyDeviceTypeInit().initHIP();
  });
}

}
