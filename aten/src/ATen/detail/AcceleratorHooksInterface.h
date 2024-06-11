#pragma once

#include <c10/core/Device.h>
#include <c10/core/Stream.h>
#include <c10/core/Storage.h>
#include <c10/core/DeviceGuard.h>
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

  virtual void getIpcHandleSize(size_t& ipc_memory_handle_size,
                                size_t& ipc_event_handle_size) const{
    TORCH_CHECK(false, "Backend doesn't support getIpcHandleSize");
  }

  virtual void StorageShareDevice(const c10::Storage& storage,
                                  ptrdiff_t& offset_bytes,
                                  std::unique_ptr<char[]>& new_memory_handle,
                                  std::unique_ptr<char[]>& new_event_handle,
                                  std::unique_ptr<char[]>& new_ref_counter,
                                  uint64_t& new_ref_counter_offset,
                                  bool& new_event_sync_required) const {
  TORCH_CHECK(false, "Backend doesn't support StorageShareDevice");
  };

  virtual void StorageNewSharedDevice(const c10::DeviceIndex& device,
                                      bool& event_sync_required,
                                      std::string& s_ipc_event_handle,
                                      std::string& s_handle,
                                      std::string& ref_counter_handle,
                                      ptrdiff_t& ref_counter_offset,
                                      ptrdiff_t& storage_offset_bytes,
                                      c10::DataPtr& data_ptr) const {
  TORCH_CHECK(false, "Backend doesn't support StorageNewSharedDevice");
  };

  virtual void getIpcRefCounterFileSize(int64_t& ipc_ref_counter_file_size) const {
    TORCH_CHECK(false, "Backend doesn't support getIpcRefCounterFileSize");
  };

};

} // namespace at
