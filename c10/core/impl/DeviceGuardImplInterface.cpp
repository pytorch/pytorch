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

// A non-null registry entry means the backend is compiled in;
// deviceCount() == 0 means no device is visible, so swap in a no-op guard.
// This lets us trace device kernels under FakeTensorMode without touching
// a physical device.
void ensureDeviceGuardSet(DeviceType type) {
  auto& slot = device_guard_impl_registry[static_cast<size_t>(type)];
  const DeviceGuardImplInterface* p = slot.load();
  if (!p || p->deviceCount() != 0) {
    return;
  }

  switch (type) {
#define CASE(D, _)                                  \
  case DeviceType::D: {                             \
    static FakeGuardImpl<DeviceType::D> fake_guard; \
    slot.store(&fake_guard);                        \
    break;                                          \
  }
    C10_FORALL_BACKEND_DEVICE_TYPES(CASE, unused)
#undef CASE
    default:
      TORCH_INTERNAL_ASSERT(false, "Unsupported device type: ", type);
  }
}

void ensureCUDADeviceGuardSet() {
  ensureDeviceGuardSet(DeviceType::CUDA);
}

} // namespace c10::impl
