#include <c10/core/impl/FakeTensorModeTLS.h>
#include <c10/core/impl/LocalDispatchKeySet.h>

namespace c10::impl {

static thread_local std::shared_ptr<FakeTensorModeHandle> fakeTensorModeState;

void FakeTensorModeTLS::set_state(std::shared_ptr<FakeTensorModeHandle> state) {
  if (state) {
    tls_set_dispatch_key_included(DispatchKey::Fake, true);
  } else {
    reset_state();
    return;
  }
  fakeTensorModeState = std::move(state);
}

void FakeTensorModeTLS::create_state(
    std::shared_ptr<FakeTensorModeHandle> state) {
  fakeTensorModeState = std::move(state);
}

std::shared_ptr<FakeTensorModeHandle> FakeTensorModeTLS::get_state() {
  return fakeTensorModeState;
}

void FakeTensorModeTLS::reset_state() {
  // Released once the slot and key are updated, as releasing the last
  // reference to a handle releases its mode's Python object, running Python
  auto state = std::move(fakeTensorModeState);
  tls_set_dispatch_key_included(DispatchKey::Fake, false);
}

static thread_local bool inKernelInvocation = false;

bool FakeTensorModeTLS::in_kernel_invocation() {
  return inKernelInvocation;
}

void FakeTensorModeTLS::set_in_kernel_invocation(bool value) {
  inKernelInvocation = value;
}

} // namespace c10::impl
