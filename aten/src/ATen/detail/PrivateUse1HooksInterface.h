#pragma once

#include <ATen/core/ATen_fwd.h>
#include <ATen/core/GeneratorForPrivateuseone.h>
#include <ATen/core/TensorBase.h>
#include <ATen/detail/AcceleratorHooksInterface.h>

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Exception.h>
#include <c10/util/OptionalArrayRef.h>


namespace at {

struct TORCH_API PrivateUse1HooksInterface : AcceleratorHooksInterface {
#define FAIL_PRIVATEUSE1HOOKS_FUNC(func)                        \
  TORCH_CHECK_NOT_IMPLEMENTED(                                  \
      false,                                                    \
      "You should register `PrivateUse1HooksInterface`",        \
      "by `RegisterPrivateUse1HooksInterface` and implement `", \
      func,                                                     \
      "` at the same time for PrivateUse1.");

  ~PrivateUse1HooksInterface() override = default;

  bool isBuilt() const override {
    FAIL_PRIVATEUSE1HOOKS_FUNC(__func__);
  }

  bool isAvailable() const override {
    FAIL_PRIVATEUSE1HOOKS_FUNC(__func__);
  }

  const at::Generator& getDefaultGenerator(
      c10::DeviceIndex /*device_index*/) const override {
    FAIL_PRIVATEUSE1HOOKS_FUNC(__func__);
  }

  Generator getNewGenerator(
      [[maybe_unused]] DeviceIndex device_index = -1) const override {
    // TODO(FFFrog): Preserved for BC and will be removed in the future.
    if (at::GetGeneratorPrivate().has_value())
      return at::GetGeneratorForPrivateuse1(device_index);

    FAIL_PRIVATEUSE1HOOKS_FUNC(__func__);
  }

  at::Device getDeviceFromPtr(void* /*data*/) const override {
    FAIL_PRIVATEUSE1HOOKS_FUNC(__func__);
  }

  bool isPinnedPtr(const void* /*data*/) const override {
    return false;
  }

  Allocator* getPinnedMemoryAllocator() const override {
    FAIL_PRIVATEUSE1HOOKS_FUNC(__func__);
  }

  bool hasPrimaryContext(DeviceIndex /*device_index*/) const override {
    FAIL_PRIVATEUSE1HOOKS_FUNC(__func__);
  }

  void init() const override {}
  virtual void resizePrivateUse1Bytes(
      const c10::Storage& /*storage*/,
      size_t /*newsize*/) const {
    FAIL_PRIVATEUSE1HOOKS_FUNC(__func__);
  }

  // Opt-in hook allowing a PrivateUse1 backend to take over Tensor/Storage
  // construction inside `at::from_blob(...)`, e.g. to produce a backend
  // subclass of TensorImpl/StorageImpl instead of the generic ones. Backends
  // that don't need this (the common case) should leave both methods at
  // their defaults below, which preserves the exact pre-existing behavior.
  virtual bool hasCustomFromBlob() const {
    return false;
  }

  virtual at::TensorBase fromBlobPrivateUse1(
      // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
      c10::DataPtr&& data_ptr,
      std::size_t size_bytes,
      at::IntArrayRef sizes,
      at::OptionalIntArrayRef strides,
      std::optional<int64_t> storage_offset,
      const at::TensorOptions& options,
      bool resizeable,
      c10::Allocator* allocator) const {
    FAIL_PRIVATEUSE1HOOKS_FUNC(__func__);
  }

#undef FAIL_PRIVATEUSE1HOOKS_FUNC
};

struct TORCH_API PrivateUse1HooksArgs {};

TORCH_API void RegisterPrivateUse1HooksInterface(
    at::PrivateUse1HooksInterface* hook_);

TORCH_API bool isPrivateUse1HooksRegistered();

namespace detail {

TORCH_API const at::PrivateUse1HooksInterface& getPrivateUse1Hooks();

} // namespace detail

} // namespace at
