#include <c10/core/impl/FakeTensorModeTLS.h>
#include <c10/core/impl/LocalDispatchKeySet.h>

namespace c10::impl {

static thread_local std::shared_ptr<FakeTensorMode> fakeTensorModeState;

void FakeTensorModeTLS::set_state(std::shared_ptr<FakeTensorMode> state) {
  if (state) {
    tls_set_dispatch_key_included(DispatchKey::Fake, true);
  } else {
    reset_state();
    return;
  }
  fakeTensorModeState = std::move(state);
}

void FakeTensorModeTLS::create_state(std::shared_ptr<FakeTensorMode> state) {
  fakeTensorModeState = std::move(state);
}

void FakeTensorModeTLS::activate() {
  TORCH_INTERNAL_ASSERT(
      fakeTensorModeState, "activate() called with no FakeTensorMode state");
  tls_set_dispatch_key_included(DispatchKey::Fake, true);
}

void FakeTensorModeTLS::deactivate() {
  TORCH_INTERNAL_ASSERT(
      fakeTensorModeState, "deactivate() called with no FakeTensorMode state");
  tls_set_dispatch_key_included(DispatchKey::Fake, false);
}

std::shared_ptr<FakeTensorMode> FakeTensorModeTLS::get_state() {
  return fakeTensorModeState;
}

void FakeTensorModeTLS::reset_state() {
  fakeTensorModeState = nullptr;
  tls_set_dispatch_key_included(DispatchKey::Fake, false);
}

static NormalizeFakeDeviceFn normalizeFakeDeviceFn = nullptr;

void setNormalizeFakeDeviceFn(NormalizeFakeDeviceFn fn) {
  normalizeFakeDeviceFn = fn;
}

DeviceIndex normalizeFakeDevice(DeviceType type) {
  return normalizeFakeDeviceFn ? normalizeFakeDeviceFn(type)
                               : static_cast<DeviceIndex>(0);
}

} // namespace c10::impl
