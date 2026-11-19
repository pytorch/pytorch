#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/impl/FakeGuardImpl.h>
#include <array>

namespace c10::impl {

std::array<
    std::atomic<const DeviceGuardImplInterface*>,
    static_cast<size_t>(DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES)>
    device_guard_impl_registry;

void registerDeviceGuard(
    DeviceType type,
    const DeviceGuardImplInterface* impl) {
  device_guard_impl_registry[static_cast<size_t>(type)].store(impl);
}

DeviceGuardImplRegistrar::DeviceGuardImplRegistrar(
    DeviceType type,
    const DeviceGuardImplInterface* impl) {
  registerDeviceGuard(type, impl);
}

namespace {

// A non-null registry entry means the backend is compiled in;
// deviceCount() == 0 means no device is visible, so swap in a no-op guard.
// This lets us trace device kernels under FakeTensorMode without touching
// a physical device.
template <DeviceType T>
void ensureDeviceGuardSet() {
  constexpr std::size_t idx = static_cast<std::size_t>(T);
  const DeviceGuardImplInterface* p = device_guard_impl_registry[idx].load();
  if (p && p->deviceCount() == 0) {
    static FakeGuardImpl<T> fake_guard;
    device_guard_impl_registry[idx].store(&fake_guard);
  }
}

} // namespace

void ensureCUDADeviceGuardSet() {
  ensureDeviceGuardSet<DeviceType::CUDA>();
}

void ensureXPUDeviceGuardSet() {
  ensureDeviceGuardSet<DeviceType::XPU>();
}

} // namespace c10::impl
