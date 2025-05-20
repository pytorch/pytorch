#pragma once

#include <ATen/core/Generator.h>

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/Stream.h>

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-parameter")

namespace at {

// AcceleratorHooksInterface is a shared interface provided by all
// accelerators to allow generic code.
// This inferface is hook-based as it corresponds to all the functions
// that are going to be called in a generic way from the CPU code.

struct TORCH_API AcceleratorHooksInterface {
  // This should never actually be implemented, but it is used to
  // squelch -Werror=non-virtual-dtor
  virtual ~AcceleratorHooksInterface() = default;

  // Whether this backend was enabled at compilation time.
  // This function should NEVER throw.
  virtual bool isBuilt() const {
    return false;
  }

  // Whether this backend can be used at runtime, meaning it was built,
  // its runtime dependencies are available (driver) and at least one
  // supported device can be used.
  // This function should NEVER throw. This function should NOT initialize the context
  // on any device (result of hasPrimaryContext below should not change).
  // While it is acceptable for this function to poison fork, it is
  // recommended to avoid doing so whenever possible.
  virtual bool isAvailable() const {
    return false;
  }

  // Whether the device at device_index is fully initialized or not.
  virtual bool hasPrimaryContext(DeviceIndex device_index) const = 0;

  virtual void init() const {
    TORCH_CHECK(false, "Backend doesn`t support init()");
  }

  virtual DeviceIndex deviceCount() const {
    return 0;
  }

  virtual void setCurrentDevice(DeviceIndex device) const {
    TORCH_CHECK(false, "Backend doesn't support setCurrentDevice()");
  }

  virtual DeviceIndex getCurrentDevice() const {
    TORCH_CHECK(false, "Backend doesn't support getCurrentDevice()");
    return -1;
  }

  virtual DeviceIndex exchangeDevice(DeviceIndex device) const {
    TORCH_CHECK(false, "Backend doesn't support exchangeDevice()");
    return -1;
  }

  virtual DeviceIndex maybeExchangeDevice(DeviceIndex device) const {
    TORCH_CHECK(false, "Backend doesn't support maybeExchangeDevice()");
    return -1;
  }

  virtual bool isPinnedPtr(const void* data) const {
    return false;
  }

  virtual Allocator* getPinnedMemoryAllocator() const {
    TORCH_CHECK(false, "Backend doesn't support getPinnedMemoryAllocator()");
    return nullptr;
  }

  virtual Device getDeviceFromPtr(void* data) const {
    TORCH_CHECK(false, "Backend doesn't support getDeviceFromPtr()");
  }

  virtual const Generator& getDefaultGenerator(
      [[maybe_unused]] DeviceIndex device_index = -1) const {
    TORCH_CHECK(false, "Backend doesn`t support getDefaultGenerator()");
  }

  virtual Generator getNewGenerator(
      [[maybe_unused]] DeviceIndex device_index = -1) const {
    TORCH_CHECK(false, "Backend doesn`t support getNewGenerator()");
  }
};

} // namespace at

C10_DIAGNOSTIC_POP()
