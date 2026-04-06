#include <gtest/gtest.h>

#include <c10/core/DeviceGuard.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/impl/FakeGuardImpl.h>

#include <thread>
#include <vector>

using namespace c10;
using namespace c10::impl;

// The tests here are mostly covered by InlineDeviceGuard_test, but there
// is some DeviceGuard specific functionality we must test.

// -- DeviceGuard -------------------------------------------------------

TEST(DeviceGuard, ResetDeviceDifferentDeviceType) {
  FakeGuardImpl<DeviceType::CUDA> cuda_impl;
  FakeGuardImpl<DeviceType::HIP> hip_impl;
  FakeGuardImpl<DeviceType::CUDA>::setDeviceIndex(0);
  FakeGuardImpl<DeviceType::HIP>::setDeviceIndex(0);
  DeviceGuard g(Device(DeviceType::CUDA, 1), &cuda_impl);
  g.reset_device(Device(DeviceType::HIP, 2), &hip_impl);
  ASSERT_EQ(FakeGuardImpl<DeviceType::CUDA>::getDeviceIndex(), 0);
  ASSERT_EQ(FakeGuardImpl<DeviceType::HIP>::getDeviceIndex(), 2);
  ASSERT_EQ(g.current_device(), Device(DeviceType::HIP, 2));
  ASSERT_EQ(g.original_device(), Device(DeviceType::HIP, 0));
}

// -- OptionalDeviceGuard -----------------------------------------------

TEST(OptionalDeviceGuard, ResetDeviceDifferentDeviceType) {
  FakeGuardImpl<DeviceType::CUDA> cuda_impl;
  FakeGuardImpl<DeviceType::HIP> hip_impl;
  FakeGuardImpl<DeviceType::CUDA>::setDeviceIndex(0);
  FakeGuardImpl<DeviceType::HIP>::setDeviceIndex(0);
  OptionalDeviceGuard g;
  g.reset_device(Device(DeviceType::CUDA, 1), &cuda_impl);
  g.reset_device(Device(DeviceType::HIP, 2), &hip_impl);
  ASSERT_EQ(FakeGuardImpl<DeviceType::CUDA>::getDeviceIndex(), 0);
  ASSERT_EQ(FakeGuardImpl<DeviceType::HIP>::getDeviceIndex(), 2);
  ASSERT_EQ(g.current_device(), Device(DeviceType::HIP, 2));
  ASSERT_EQ(g.original_device(), Device(DeviceType::HIP, 0));
}

// -- ensureCUDADeviceGuardSet -------------------------------------------

// Regression test: ensureCUDADeviceGuardSet() used to store a thread-local
// FakeGuardImpl* in the global device_guard_impl_registry.  When the owning
// thread exited its TLS was freed, leaving a dangling pointer that the next
// thread to call deviceCount() would dereference (segfault).
//
// The fix is a function-local static, which has program lifetime.  We verify
// that the pointer in the registry is still valid (and returns the expected
// deviceCount) after the threads that triggered guard installation have exited.
TEST(EnsureCUDADeviceGuard, NoUseAfterFreeWhenThreadsExit) {
  // Simulate "CUDA compiled, no devices visible": a guard that is non-null but
  // returns deviceCount() == 0, which is the condition that triggers fake guard
  // installation in ensureCUDADeviceGuardSet().
  struct ZeroDeviceGuardImpl final : public DeviceGuardImplInterface {
    DeviceType type() const override {
      return DeviceType::CUDA;
    }
    Device exchangeDevice(Device d) const override {
      return d;
    }
    Device getDevice() const override {
      return Device(DeviceType::CUDA, 0);
    }
    void setDevice(Device) const override {}
    void uncheckedSetDevice(Device) const noexcept override {}
    Stream getStream(Device d) const noexcept override {
      return Stream(Stream::UNSAFE, d, 0);
    }
    Stream exchangeStream(Stream s) const noexcept override {
      return s;
    }
    DeviceIndex deviceCount() const noexcept override {
      return 0;
    }
    void record(void**, const Stream&, const DeviceIndex, const EventFlag)
        const override {}
    void block(void*, const Stream&) const override {}
    bool queryEvent(void*) const override {
      return true;
    }
    void destroyEvent(void*, const DeviceIndex) const noexcept override {}
  };

  constexpr auto cuda_idx = static_cast<size_t>(DeviceType::CUDA);
  const auto* saved = device_guard_impl_registry[cuda_idx].load();

  static ZeroDeviceGuardImpl zero_impl;
  device_guard_impl_registry[cuda_idx].store(&zero_impl);

  // Phase 1: threads call ensureCUDADeviceGuardSet(), detect deviceCount()==0,
  // and install a FakeGuardImpl in the global registry.
  {
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; i++) {
      threads.emplace_back(ensureCUDADeviceGuardSet);
    }
    for (auto& t : threads) {
      t.join();
    }
  }
  // The threads' TLS is now destroyed.  With the old code the registry now
  // holds a dangling pointer; with the fix it holds &fake_cuda_guard (static).

  // Phase 2: the pointer must still be valid and return the expected count.
  const auto* p = device_guard_impl_registry[cuda_idx].load();
  ASSERT_NE(p, nullptr);
  ASSERT_EQ(p->deviceCount(), kFakeGuardImplMaxDevices);

  device_guard_impl_registry[cuda_idx].store(saved);
}
