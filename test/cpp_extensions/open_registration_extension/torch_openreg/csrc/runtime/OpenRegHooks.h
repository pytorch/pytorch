#pragma once

#include <ATen/core/CachingHostAllocator.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>

#include <include/openreg.h>

#include "OpenRegFunctions.h"
#include "OpenRegGenerator.h"

namespace c10::openreg {
struct OPENREG_EXPORT OpenRegHooksInterface : public at::PrivateUse1HooksInterface {
  OpenRegHooksInterface() {};
  ~OpenRegHooksInterface() override = default;

  bool hasPrimaryContext(DeviceIndex device_index) const override {
    return true;
  }

  bool isBuilt() const override {
    // This extension is compiled as part of the OpenReg test extension.
    return true;
  }

  bool isAvailable() const override {
    // Consider OpenReg available if there's at least one device reported.
    return device_count() > 0;
  }

  DeviceIndex deviceCount() const override {
    return device_count();
  }

  void setCurrentDevice(DeviceIndex device) const override {
    set_device(device);
  }

  DeviceIndex getCurrentDevice() const override {
    return current_device();
  }

  DeviceIndex exchangeDevice(DeviceIndex device) const override {
    return ExchangeDevice(device);
  }

  DeviceIndex maybeExchangeDevice(DeviceIndex device) const override {
    // Only exchange if the requested device is valid; otherwise, no-op and return current
    auto count = device_count();
    if (device < 0 || device >= count) {
      return getCurrentDevice();
    }
    return exchangeDevice(device);
  }

  at::Allocator* getPinnedMemoryAllocator() const override {
    return at::getHostAllocator(at::kPrivateUse1);
  }

  bool isPinnedPtr(const void* data) const override {
    orPointerAttributes attr{};
    orPointerGetAttributes(&attr, data);

    return attr.type == orMemoryTypeHost;
  }

  at::Device getDeviceFromPtr(void* data) const override {
    orPointerAttributes attr{};
    auto err = orPointerGetAttributes(&attr, data);
    if (err == orSuccess && attr.type == orMemoryTypeDevice) {
      return at::Device(at::DeviceType::PrivateUse1, static_cast<int>(attr.device));
    } else {
      TORCH_CHECK(false, "failed to get device from pointer");
    }
    return at::Device(at::DeviceType::PrivateUse1, current_device());
  }
  // LITERALINCLUDE START: OPENREG HOOK EXAMPLES
  const at::Generator& getDefaultGenerator(DeviceIndex device_index) const override {
    return getDefaultOpenRegGenerator(device_index);
  }
  // LITERALINCLUDE END: OPENREG HOOK EXAMPLES

  at::Generator getNewGenerator(DeviceIndex device_index) const override {
    return at::make_generator<OpenRegGeneratorImpl>(device_index);
  }
};

} // namespace c10::openreg
