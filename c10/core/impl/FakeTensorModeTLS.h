#pragma once
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/macros/Export.h>

#include <memory>

namespace c10 {
struct FakeTensorMode;
} // namespace c10

namespace c10::impl {

class C10_API FakeTensorModeTLS {
 public:
  static void set_state(std::shared_ptr<FakeTensorMode> state);
  static void create_state(std::shared_ptr<FakeTensorMode> state);
  static std::shared_ptr<FakeTensorMode> get_state();
  static void reset_state();

  // Note [in_kernel_invocation]
  // A Python FakeTensor reports its fake device, except inside
  // FakeTensorMode's in_kernel_invocation_manager, where a meta kernel is
  // running on it and must see the meta device. A C++ fake stores its fake
  // device in ExtraMeta, and TensorImpl::device_custom reports meta while this
  // thread-local flag is set. Excluding DispatchKey::Fake does not change the
  // reported device, so code can suspend fake dispatch without fakes changing
  // device.
  static bool in_kernel_invocation();
  static void set_in_kernel_invocation(bool value);
};

struct C10_API FakeInKernelInvocationGuard {
  FakeInKernelInvocationGuard()
      : prev_(FakeTensorModeTLS::in_kernel_invocation()) {
    FakeTensorModeTLS::set_in_kernel_invocation(true);
  }
  ~FakeInKernelInvocationGuard() {
    FakeTensorModeTLS::set_in_kernel_invocation(prev_);
  }
  FakeInKernelInvocationGuard(const FakeInKernelInvocationGuard&) = delete;
  FakeInKernelInvocationGuard& operator=(const FakeInKernelInvocationGuard&) =
      delete;
  FakeInKernelInvocationGuard(FakeInKernelInvocationGuard&&) = delete;
  FakeInKernelInvocationGuard& operator=(FakeInKernelInvocationGuard&&) =
      delete;

 private:
  bool prev_;
};

// Installs mode (or no mode, for nullptr) in TLS with the Fake key to match;
// the destructor restores the previous mode and Fake key state exactly.
struct C10_API FakeTensorModeGuard {
  explicit FakeTensorModeGuard(std::shared_ptr<FakeTensorMode> mode)
      : prev_mode_(FakeTensorModeTLS::get_state()),
        prev_fake_included_(tls_is_dispatch_key_included(DispatchKey::Fake)) {
    FakeTensorModeTLS::set_state(std::move(mode));
  }
  ~FakeTensorModeGuard() {
    FakeTensorModeTLS::create_state(std::move(prev_mode_));
    tls_set_dispatch_key_included(DispatchKey::Fake, prev_fake_included_);
  }
  FakeTensorModeGuard(const FakeTensorModeGuard&) = delete;
  FakeTensorModeGuard& operator=(const FakeTensorModeGuard&) = delete;
  FakeTensorModeGuard(FakeTensorModeGuard&&) = delete;
  FakeTensorModeGuard& operator=(FakeTensorModeGuard&&) = delete;

 private:
  std::shared_ptr<FakeTensorMode> prev_mode_;
  bool prev_fake_included_;
};

} // namespace c10::impl
