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

void init_cpp_fake(
    c10::TensorImpl* impl,
    c10::Device device,
    const std::shared_ptr<c10::FakeTensorMode>& mode) {
  TORCH_INTERNAL_ASSERT(mode != nullptr);
  // Both checked before impl is changed; once its fake device is set, impl no
  // longer reads as meta
  TORCH_CHECK(
      impl->is_meta(),
      "init_cpp_fake expects a meta tensor, got a tensor on ",
      impl->device());
  TORCH_CHECK(
      device.type() != c10::kMeta || mode->allow_meta_,
      "device.type must not be 'meta' when allow_meta is False");
  set_and_normalize_fake_device(impl, device);
  impl->set_fake_tensor_mode(mode);
  if (!impl->has_storage()) {
    return;
  }
  auto* storage_impl = impl->storage().unsafeGetStorageImpl();
  if (mode->allow_unsafe_data_ptr_access_) {
    storage_impl->set_warn_deprecated_on_mutable_data_ptr();
  } else {
    storage_impl->set_throw_on_mutable_data_ptr();
  }
}

} // namespace at
