#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <array>

namespace c10::impl {

std::array<
    std::atomic<const DeviceGuardImplInterface*>,
    static_cast<size_t>(DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES)>
    device_guard_impl_registry;

DeviceGuardImplRegistrar::DeviceGuardImplRegistrar(
    DeviceType type,
    const DeviceGuardImplInterface* impl) {
  device_guard_impl_registry[static_cast<size_t>(type)].store(impl);
}

namespace {
thread_local std::unique_ptr<DeviceGuardImplInterface> tls_fake_device_guard =
    nullptr;
}

void ensureCUDADeviceGuardSet() {
  constexpr auto cuda_idx = static_cast<std::size_t>(DeviceType::CUDA);

  const DeviceGuardImplInterface* p =
      device_guard_impl_registry[cuda_idx].load();

  // A non-null `ptr` indicates that CUDA is already available.
  if (p == nullptr || (p && p->deviceCount() == 0)) {
    // In following cases, we override CUDA guard interface with a no-op
    // device guard.
    // 1. p == nullptr; Trying to get a cuda device guard on a cpu-only build.
    // 2. p->deviceCount() == 0; cuda build enabled, but no cuda devices
    // available.
    tls_fake_device_guard =
        std::make_unique<NoOpDeviceGuardImpl<DeviceType::CUDA>>();
    device_guard_impl_registry[cuda_idx].store(tls_fake_device_guard.get());
  }
}

} // namespace c10::impl
