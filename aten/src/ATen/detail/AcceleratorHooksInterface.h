#pragma once

#include <c10/core/Device.h>
#include <c10/core/Stream.h>
#include <c10/core/Allocator.h>
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

  virtual c10::Stream getCurrentStream(C10_UNUSED DeviceIndex device) const {
    TORCH_CHECK(false, "Backend doesn't support getCurrentStream()");
    return c10::Stream::unpack3(-1, 0, c10::DeviceType::CPU);
  }

  // NB: Not all backends support getDefaultStream(), such as Intel XPU,
  // because the stream may have an implicit synchronization semantic.
  // Ignore it if your backend does not support this feature.
  virtual c10::Stream getDefaultStream(C10_UNUSED DeviceIndex device) const {
    TORCH_CHECK(false, "Backend doesn't support getDefaultStream()");
    return c10::Stream::unpack3(-1, 0, c10::DeviceType::CPU);
  }

  virtual void setCurrentStream(C10_UNUSED const c10::Stream& stream) {
    TORCH_CHECK(false, "Backend doesn't support setCurrentStream()");
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
};

} // namespace at
C10_DIAGNOSTIC_POP()
