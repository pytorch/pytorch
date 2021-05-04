#include <c10/core/impl/DeviceGuardImplInterface.h>

namespace c10 {
namespace impl {

// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
std::atomic<const DeviceGuardImplInterface*>
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    device_guard_impl_registry[static_cast<size_t>(
        DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES)];

DeviceGuardImplRegistrar::DeviceGuardImplRegistrar(
    DeviceType type,
    const DeviceGuardImplInterface* impl) {
  device_guard_impl_registry[static_cast<size_t>(type)].store(impl);
}

} // namespace impl
} // namespace c10
