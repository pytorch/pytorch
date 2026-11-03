#include <ATen/FakeTensor.h>

#include <ATen/Context.h>
#include <c10/core/TensorImpl.h>

namespace at {
namespace {

bool isIndexedDeviceType(c10::DeviceType type) {
  return type == kCUDA || type == kHIP || type == kHPU || type == kXPU ||
      type == kMPS || type == kMTIA || type == kPrivateUse1;
}

c10::Device normalizeFakeDevice(c10::Device device) {
  if (device.has_index() || !isIndexedDeviceType(device.type())) {
    return device;
  }

  const auto type = device.type();
  auto& context = globalContext();
  const auto& hooks = context.getAcceleratorHooksInterface(type);
  if (type == kMPS || !hooks.isBuilt()) {
    return c10::Device(type, 0);
  }
  for (c10::DeviceIndex i = 0; i < hooks.deviceCount(); i++) {
    if (hooks.hasPrimaryContext(i)) {
      return c10::Device(type, hooks.getCurrentDevice());
    }
  }
  return c10::Device(type, 0);
}

} // namespace

void set_and_normalize_fake_device(
    c10::TensorImpl* impl,
    c10::Device device) {
  auto fake_device = normalizeFakeDevice(device);
  TORCH_INTERNAL_ASSERT(
      !isIndexedDeviceType(fake_device.type()) || fake_device.has_index());
  impl->set_fake_device(fake_device);
}

void mark_fake_storage(
    c10::TensorImpl* impl,
    const c10::FakeTensorMode& mode) {
  if (!impl->has_storage()) {
    return;
  }
  auto* storage_impl = impl->storage().unsafeGetStorageImpl();
  if (mode.allow_unsafe_data_ptr_access_) {
    storage_impl->set_warn_deprecated_on_mutable_data_ptr();
  } else {
    storage_impl->set_throw_on_mutable_data_ptr();
  }
}

} // namespace at
