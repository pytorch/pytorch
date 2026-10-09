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

// Note [in_kernel_invocation]
// A Python FakeTensor reports its fake device, except inside FakeTensorMode's
// in_kernel_invocation_manager, where a meta kernel is running on it and must
// see the meta device. A C++ fake stores its fake device in ExtraMeta, and
// TensorImpl::device_custom reports meta while this thread-local flag is set.
// Excluding DispatchKey::Fake does not change the reported device, so code can
// suspend fake dispatch without fakes changing device.
C10_API bool in_kernel_invocation();
C10_API void set_in_kernel_invocation(bool value);

// Sets the flag for the guard's lifetime and restores the previous value,
// including when the kernel throws.
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

} // namespace c10::impl
