#include <c10/core/impl/DeviceGuardImplInterface.h>

namespace c10 {
namespace impl {

std::atomic<const DeviceGuardImplInterface*>
device_guard_impl_registry[static_cast<size_t>(DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES)];

DeviceGuardImplRegistrar::DeviceGuardImplRegistrar(DeviceType type, const DeviceGuardImplInterface* impl) {
  device_guard_impl_registry[static_cast<size_t>(type)].store(impl);
}

}} // namespace c10::impl
