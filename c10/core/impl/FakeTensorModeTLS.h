#pragma once
#include <c10/core/TensorImpl.h>
#include <c10/macros/Export.h>

namespace c10::impl {

class C10_API FakeTensorModeTLS {
 public:
  static void set_state(std::shared_ptr<FakeTensorMode> state);
  static void create_state(std::shared_ptr<FakeTensorMode> state);
  static std::shared_ptr<FakeTensorMode> get_state();
  static void reset_state();
};

// [in_kernel_invocation] flag: same as Python FakeTensorMode's
// in_kernel_invocation manager. Signals whether a meta kernel is currently
// active; if so, a fake tensor must return its meta device.
C10_API bool in_kernel_invocation();
C10_API void set_in_kernel_invocation(bool value);

// RAII so the flag is restored even if the meta kernel throws (e.g. no meta
// kernel is found and the fallback path runs).
struct C10_API FakeInKernelInvocationGuard {
  FakeInKernelInvocationGuard() : prev_(in_kernel_invocation()) {
    set_in_kernel_invocation(true);
  }
  ~FakeInKernelInvocationGuard() {
    set_in_kernel_invocation(prev_);
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

// Resolves an unindexed fake device (e.g. "cuda") to a concrete index the same
// way python's FakeTensor._normalize_fake_device does: the backend's current
// device if it is already initialized, else 0.
using NormalizeFakeDeviceFn = DeviceIndex (*)(DeviceType);
C10_API void setNormalizeFakeDeviceFn(NormalizeFakeDeviceFn fn);
C10_API DeviceIndex normalizeFakeDevice(DeviceType type);

} // namespace c10::impl
