#include <ATen/FakeTensor.h>

#include <ATen/Context.h>
#include <ATen/core/Tensor.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/impl/FakeTensorModeTLS.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/impl/TorchDispatchModeTLS.h>

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

Tensor fake_shallow_copy(
    const Tensor& self,
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) {
  TORCH_INTERNAL_ASSERT(!c10::impl::FakeTensorModeTLS::in_kernel_invocation());
  return Tensor(self.getIntrusivePtr()->shallow_copy_and_detach(
      version_counter, allow_tensor_metadata_change));
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

namespace impl {

bool may_turn_const(const Tensor& t) {
  return t.numel() <= 1 && !t.is_fake() && !t.is_sparse() &&
      !t.unsafeGetTensorImpl()->is_sparse_compressed() && !t.device().is_meta();
}

Tensor fake_detach(const Tensor& self) {
  return fake_shallow_copy(
      self, /*version_counter=*/0, /*allow_tensor_metadata_change=*/false);
}

Tensor fake_tensor_data(
    const Tensor& self,
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) {
  const bool python_hook =
      !c10::impl::tls_is_dispatch_key_excluded(c10::DispatchKey::Python) &&
      (c10::impl::TorchDispatchModeTLS::stack_len() > 0 ||
       self.key_set().has(c10::DispatchKey::Python));
  auto result =
      fake_shallow_copy(self, version_counter, allow_tensor_metadata_change);
  if (!python_hook) {
    if (auto real = self.unsafeGetTensorImpl()->real_tensor()) {
      result.unsafeGetTensorImpl()->set_real_tensor(real->shallow_copy_and_detach(
          real->version_counter(), /*allow_tensor_metadata_change=*/false));
    }
    auto mode = self.unsafeGetTensorImpl()->fake_tensor_mode();
    if (auto constant = mode->get_constant(self.unsafeGetTensorImpl())) {
      // ExtraMeta's shallow copy drops constants; register this storage alias.
      mode->set_constant(
          result.unsafeGetTensorImpl(),
          constant->shallow_copy_and_detach(
              constant->version_counter(),
              /*allow_tensor_metadata_change=*/false));
    }
  }
  return result;
}

Tensor& fake_shallow_copy_data_(Tensor& self, const Tensor& source) {
  TORCH_INTERNAL_ASSERT(!c10::impl::FakeTensorModeTLS::in_kernel_invocation());
  // Keep set_data's type/dtype checks and gradient accumulator reset.
  self.set_data(source);
  return self;
}

} // namespace impl

} // namespace at
