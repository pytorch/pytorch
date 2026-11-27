#include <c10/core/impl/FakeTensorModeTLS.h>
#include <c10/core/impl/LocalDispatchKeySet.h>

#include <utility>

namespace c10::impl {

namespace {

thread_local bool fakeTensorModeTLSClosed = false;

struct FakeTensorModeTLSState {
  FakeTensorModeTLSState() = default;
  FakeTensorModeTLSState(const FakeTensorModeTLSState&) = delete;
  FakeTensorModeTLSState& operator=(const FakeTensorModeTLSState&) = delete;
  FakeTensorModeTLSState(FakeTensorModeTLSState&&) = delete;
  FakeTensorModeTLSState& operator=(FakeTensorModeTLSState&&) = delete;

  std::shared_ptr<FakeTensorMode> state;

  ~FakeTensorModeTLSState() {
    // Last release can run Python, including lookups from weakref callbacks.
    // Only the fake slot is protected; other dispatch TLS may be destroyed.
    fakeTensorModeTLSClosed = true;
    auto saved = std::move(state);
    tls_set_dispatch_key_included(DispatchKey::Fake, false);
  }
};

thread_local FakeTensorModeTLSState fakeTensorModeState;

} // namespace

void FakeTensorModeTLS::set_state(std::shared_ptr<FakeTensorMode> state) {
  if (state) {
    TORCH_CHECK(
        !fakeTensorModeTLSClosed,
        "Cannot install a C++ FakeTensorMode during thread teardown");
    tls_set_dispatch_key_included(DispatchKey::Fake, true);
  } else {
    reset_state();
    return;
  }
  fakeTensorModeState.state = std::move(state);
}

std::shared_ptr<FakeTensorMode> FakeTensorModeTLS::create_state(
    std::shared_ptr<FakeTensorMode> state) {
  if (fakeTensorModeTLSClosed) {
    TORCH_CHECK(
        !state, "Cannot install a C++ FakeTensorMode during thread teardown");
    return nullptr;
  }
  return std::exchange(fakeTensorModeState.state, std::move(state));
}

std::shared_ptr<FakeTensorMode> FakeTensorModeTLS::get_state() {
  return fakeTensorModeTLSClosed ? nullptr : fakeTensorModeState.state;
}

void FakeTensorModeTLS::reset_state() {
  if (fakeTensorModeTLSClosed) {
    return;
  }
  // Released once the slot and key are updated, as releasing the last
  // reference to a mode can release its converter, running Python
  auto state = std::move(fakeTensorModeState.state);
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
