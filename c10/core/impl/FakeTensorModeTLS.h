#pragma once
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

  // Note [C++ fake in_kernel_invocation]
  // A Python FakeTensor reports its fake device, except inside
  // FakeTensorMode's in_kernel_invocation_manager, where a meta kernel is
  // running on it and must see the meta device (see [in_kernel_invocation] in
  // torch/_subclasses/fake_tensor.py). A C++ fake stores its fake device in
  // ExtraMeta, and TensorImpl::device_custom reports meta while this
  // thread-local flag is set; in_kernel_invocation_manager sets it too.
  // Excluding DispatchKey::Fake does not change the reported device, so code
  // can suspend fake dispatch without fakes changing device.
  //
  // Unlike the Python flag, which is a field on one FakeTensorMode, this flag
  // belongs to the thread and is not tied to a mode:
  // - While it is set, every C++ fake on the thread reports meta, whichever
  //   mode owns it, so in_kernel_invocation_manager(mode_a) also affects fakes
  //   of mode_b. Code that runs with it unset, such as to_meta_tensor in
  //   torch/csrc/PyInterpreter.cpp, sees the fake device of a fake from
  //   another mode.
  // - at::ThreadLocalState carries it into worker threads, next to the mode.
  // - reset_state() leaves it alone on purpose: the flag records that a meta
  //   kernel is running, not which mode is installed, and the
  //   FakeInKernelInvocationGuard that set it restores the previous value when
  //   its scope ends.
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

} // namespace c10::impl
