#include <c10/core/impl/DeviceGuardImplInterface.h>

namespace c10 {
namespace impl {

Stream DeviceGuardImplInterface::getDefaultStream(Device) const {
  TORCH_CHECK(false, "Backend doesn't support acquiring a default stream.")
}

void DeviceGuardImplInterface::destroyEvent (
  void* event,
  const DeviceIndex device_index) const noexcept { }

void DeviceGuardImplInterface::record(
  void** event,
  const Stream& stream,
  const DeviceIndex device_index,
  const c10::EventFlag flag) const {
  TORCH_CHECK(false, "Backend doesn't support events.");
}

void DeviceGuardImplInterface::block(
  void* event,
  const Stream& stream) const {
  TORCH_CHECK(false, "Backend doesn't support events.");
}

bool DeviceGuardImplInterface::queryEvent(void* event) const {
  TORCH_CHECK(false, "Backend doesn't support events.");
}

DeviceGuardImplInterface::~DeviceGuardImplInterface() = default;

std::atomic<const DeviceGuardImplInterface*>
device_guard_impl_registry[static_cast<size_t>(DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES)];

DeviceGuardImplRegistrar::DeviceGuardImplRegistrar(DeviceType type, const DeviceGuardImplInterface* impl) {
  device_guard_impl_registry[static_cast<size_t>(type)].store(impl);
}

}} // namespace c10::impl
