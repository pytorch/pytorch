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

std::shared_ptr<FakeTensorMode> FakeTensorModeTLS::get_state() {
  return fakeTensorModeState;
}

void FakeTensorModeTLS::reset_state() {
  fakeTensorModeState = nullptr;
  tls_set_dispatch_key_included(DispatchKey::Fake, false);
}

} // namespace c10::impl
