#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Stream.h>

namespace c10::CachingDeviceAllocator {

using namespace c10::CachingAllocator;

// Struct containing memory allocator summary statistics for a device.
struct DeviceStats {
  // COUNT: allocations requested by client code
  StatArray allocation;
  // COUNT: number of allocated segments from device memory allocation.
  StatArray segment;
  // COUNT: number of active memory blocks (allocated or used by stream)
  StatArray active;
  // COUNT: number of inactive, split memory blocks (unallocated but can't be
  // released via device memory deallocation)
  StatArray inactive_split;

  // SUM: bytes allocated by this memory alocator
  StatArray allocated_bytes;
  // SUM: bytes reserved by this memory allocator (both free and used)
  StatArray reserved_bytes;
  // SUM: bytes within active memory blocks
  StatArray active_bytes;
  // SUM: bytes within inactive, split memory blocks
  StatArray inactive_split_bytes;
  // SUM: bytes requested by client code
  StatArray requested_bytes;

  // COUNT: total number of failed calls to device malloc necessitating cache
  // flushes.
  int64_t num_alloc_retries = 0;

  // COUNT: total number of OOMs (i.e. failed calls to device memory allocation
  // after cache flush)
  int64_t num_ooms = 0;

  // COUNT: total number of oversize blocks allocated from pool
  Stat oversize_allocations;

  // COUNT: total number of oversize blocks requiring malloc
  Stat oversize_segments;

  // COUNT: total number of synchronize_and_free_events() calls
  int64_t num_sync_all_streams = 0;

  // COUNT: total number of device memory allocation calls. This includes both
  // mapped and malloced memory.
  int64_t num_device_alloc = 0;

  // COUNT: total number of device memory deallocation calls. This includes both
  // un-mapped and free memory.
  int64_t num_device_free = 0;

  // SIZE: maximum block size that is allowed to be split.
  int64_t max_split_size = 0;
};

class C10_API DeviceAllocatorInterface : public c10::Allocator {
 public:
  virtual void* raw_alloc(size_t nbytes) = 0;
  virtual void* raw_alloc_with_stream(size_t nbytes, c10::Stream stream) = 0;
  virtual void raw_delete(void* ptr) = 0;
  virtual void init(int device_count) = 0;
  virtual bool initialized() = 0;
  virtual void emptyCache() = 0;
  virtual void enable(bool value) = 0;
  virtual bool isEnabled() const = 0;
  virtual void* getBaseAllocation(void* ptr, size_t* size) = 0;
  virtual void recordStream(const DataPtr&, c10::Stream stream) = 0;
  virtual DeviceStats getDeviceStats(c10::DeviceIndex device) = 0;
  virtual void resetAccumulatedStats(c10::DeviceIndex device) = 0;
  virtual void resetPeakStats(c10::DeviceIndex device) = 0;
  virtual std::string name() = 0;
};

template <typename T>
struct CachingDeviceAllocatorInterface : public at::DeviceAllocatorInterface {
  CachingDeviceAllocatorInterface() : impl_(std::make_unique<T>()) {}

  at::DataPtr allocate(size_t size) override {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "Not implemented for allocate");
  }

  void free(void* ctx) {
    impl_->free(ctx);
  }

  template <typename S>
  bool record_event(void* ptr, void* ctx, S stream) {
    return impl_->record_event(ptr, ctx, stream);
  }

  void empty_cache() {
    impl_->empty_cache();
  }

  void copy_data(void* dest, const void* src, std::size_t count)
      const override {
    impl_->copy_data(dest, src, count);
  }

  DeviceStats getDeviceStats() {
    return impl_->getStats();
  }

  void resetAccumulatedStats() {
    impl_->resetAccumulatedStats();
  }

  void resetPeakStats() {
    impl_->resetPeakStats();
  }

  std::unique_ptr<T> impl_;
};

/**
 * Note [DeviceAllocator design]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 */

template <typename S, typename E, typename B = HostBlock<S>>
struct CachingDeviceAllocatorImpl {
  virtual ~CachingDeviceAllocatorImpl() = default;

 public:
 private:
 protected:
};

} // namespace c10::CachingDeviceAllocator
