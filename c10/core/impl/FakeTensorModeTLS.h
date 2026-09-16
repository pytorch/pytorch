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

// [in_kernel_invocation] records that a Meta kernel is interpreting fake
// tensors as their Meta backing tensors.
C10_API bool in_kernel_invocation();
C10_API void set_in_kernel_invocation(bool value);

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
