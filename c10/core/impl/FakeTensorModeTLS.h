#pragma once
#include <c10/core/TensorImpl.h>
#include <c10/macros/Export.h>

namespace c10::impl {

class C10_API FakeTensorModeTLS {
 public:
  static void set_state(std::shared_ptr<FakeTensorMode> state);
  static void create_state(std::shared_ptr<FakeTensorMode> state);
  static void activate();
  static void deactivate();
  static std::shared_ptr<FakeTensorMode> get_state();
  static void reset_state();
};

// Resolves an unindexed fake device (e.g. "cuda") to a concrete index the same
// way python's FakeTensor._normalize_fake_device does: the backend's current
// device if it is already initialized, else 0.
using NormalizeFakeDeviceFn = DeviceIndex (*)(DeviceType);
C10_API void setNormalizeFakeDeviceFn(NormalizeFakeDeviceFn fn);
C10_API DeviceIndex normalizeFakeDevice(DeviceType type);

} // namespace c10::impl
