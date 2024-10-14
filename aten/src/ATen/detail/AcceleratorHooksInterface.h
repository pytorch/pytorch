#pragma once

#include <ATen/core/Generator.h>

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/Stream.h>
#include <c10/core/Storage.h>
#include <c10/core/DeviceGuard.h>

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

  virtual std::tuple<size_t, size_t, ptrdiff_t, std::string, std::string, std::string, uint64_t, bool>
  StorageShareDevice(const c10::Storage& storage) const {
    TORCH_CHECK(false, "Backend doesn't support StorageShareDevice");
  };

  virtual c10::DataPtr StorageNewSharedDevice(c10::DeviceIndex device,
                                              bool event_sync_required,
                                              std::string s_ipc_event_handle,
                                              std::string s_handle,
                                              std::string ref_counter_handle,
                                              ptrdiff_t ref_counter_offset,
                                              ptrdiff_t storage_offset_bytes) const {
    TORCH_CHECK(false, "Backend doesn't support StorageNewSharedDevice");
  };

  virtual int64_t getIpcRefCounterFileSize() const {
    TORCH_CHECK(false, "Backend doesn't support getIpcRefCounterFileSize");
    return -1;
  };
};

} // namespace at

C10_DIAGNOSTIC_POP()