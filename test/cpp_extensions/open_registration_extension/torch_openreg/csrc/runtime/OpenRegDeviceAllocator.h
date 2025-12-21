#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/CachingDeviceAllocator.h>
#include <c10/core/Device.h>
#include <c10/util/flat_hash_map.h>

#include <include/openreg.h>

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace c10::openreg {

class DeviceMemoryAllocator {
 public:
  explicit DeviceMemoryAllocator(c10::DeviceIndex device_index);

  DeviceMemoryAllocator(const DeviceMemoryAllocator&) = delete;
  DeviceMemoryAllocator& operator=(const DeviceMemoryAllocator&) = delete;

  void* malloc(size_t nbytes);

  void free(void* ptr);

  c10::CachingDeviceAllocator::DeviceStats getStats();

  void resetAccumulatedStats();

  void resetPeakStats();

 private:
  c10::DeviceIndex device_index_;

  c10::CachingDeviceAllocator::DeviceStats stats_;

  std::unordered_map<void*, size_t> allocation_sizes_;

  std::recursive_mutex mutex_;
};


class OpenRegDeviceAllocator final : public c10::DeviceAllocator {
 public:
  OpenRegDeviceAllocator();

  at::DataPtr allocate(size_t nbytes) override;
  at::DeleterFnPtr raw_deleter() const override;
  void copy_data(void* dest, const void* src, std::size_t count) const final;


  bool initialized() override;
  void emptyCache(MempoolId_t mempool_id = {0, 0}) override;
  void recordStream(const DataPtr& ptr, c10::Stream stream) override;
  c10::CachingDeviceAllocator::DeviceStats getDeviceStats(
      c10::DeviceIndex device) override;
  void resetAccumulatedStats(c10::DeviceIndex device) override;
  void resetPeakStats(c10::DeviceIndex device) override;


  void freeMemory(void* ptr);

 private:

  // Per-device allocators
  std::vector<std::unique_ptr<DeviceMemoryAllocator>> device_allocators_;

  // Global mapping from pointer to device index
  std::recursive_mutex mutex_;
  ska::flat_hash_map<void*, c10::DeviceIndex> allocated_blocks_;
};




}
